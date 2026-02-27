"""Gap report generation and formatting.

Produces structured reports from :class:`~agent_sim_bridge.gap.estimator.GapEstimator`
results in three output formats: plain text, JSON, and Markdown.

Classes
-------
GapReport
    Frozen dataclass holding all report fields.
GapReporter
    Generates and formats gap reports.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from agent_sim_bridge.gap.estimator import DimensionGap, GapDimension, GapEstimator, GapMetric

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GapReport:
    """Structured output of a full sim-to-real gap analysis.

    Attributes
    ----------
    report_id:
        UUID string uniquely identifying this report.
    created_at:
        UTC timestamp when the report was generated.
    dimensions:
        Mapping from dimension name to its list of :class:`DimensionGap` results.
    overall_score:
        Weighted average gap score in [0, 1] (lower is better).
    summary:
        One-sentence human-readable summary of the overall gap level.
    recommendations:
        Ordered list of actionable improvement suggestions.
    """

    report_id: str
    created_at: datetime
    dimensions: dict[str, list[DimensionGap]]
    overall_score: float
    summary: str
    recommendations: list[str]


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


class GapReporter:
    """Generate and format sim-to-real gap reports.

    Example
    -------
    ::

        estimator = GapEstimator()
        dim = GapDimension(
            name="torque",
            sim_distribution=[0.1, 0.3, 0.4, 0.2],
            real_distribution=[0.15, 0.25, 0.35, 0.25],
        )
        results = estimator.estimate_all([dim])
        reporter = GapReporter()
        report = reporter.generate_report(results, [dim])
        print(reporter.format_text(report))
    """

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def generate_report(
        self,
        estimator_results: dict[str, list[DimensionGap]],
        dimensions: list[GapDimension],
    ) -> GapReport:
        """Build a :class:`GapReport` from estimator output.

        Parameters
        ----------
        estimator_results:
            Output of :meth:`~GapEstimator.estimate_all`.
        dimensions:
            The original :class:`GapDimension` list (used to build the
            estimator on-demand for the overall score).

        Returns
        -------
        GapReport
            Fully populated report.
        """
        estimator = GapEstimator()
        overall_score = estimator.overall_gap_score(estimator_results)
        summary = self._generate_summary(overall_score)
        recommendations = self._generate_recommendations(estimator_results)

        report = GapReport(
            report_id=str(uuid.uuid4()),
            created_at=datetime.now(tz=timezone.utc),
            dimensions=estimator_results,
            overall_score=overall_score,
            summary=summary,
            recommendations=recommendations,
        )
        logger.info(
            "Gap report %s generated: score=%.4f, dimensions=%d",
            report.report_id,
            overall_score,
            len(estimator_results),
        )
        return report

    def format_text(self, report: GapReport) -> str:
        """Render a human-readable plain-text report.

        Parameters
        ----------
        report:
            The :class:`GapReport` to format.

        Returns
        -------
        str
            Multi-line text suitable for terminal output.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("Sim-to-Real Gap Report")
        lines.append("=" * 60)
        lines.append(f"Report ID   : {report.report_id}")
        lines.append(f"Created At  : {report.created_at.isoformat()}")
        lines.append(f"Overall Score: {report.overall_score:.4f} (0=no gap, 1=maximum gap)")
        lines.append("")
        lines.append(f"Summary: {report.summary}")
        lines.append("")

        if report.dimensions:
            lines.append("Dimension Results:")
            lines.append("-" * 60)
            for dimension_name, gaps in sorted(report.dimensions.items()):
                lines.append(f"  {dimension_name}:")
                for gap in gaps:
                    lines.append(
                        f"    {gap.metric.value:<20} {gap.value:>10.6f}  [{gap.interpretation}]"
                    )
            lines.append("")

        if report.recommendations:
            lines.append("Recommendations:")
            for i, recommendation in enumerate(report.recommendations, start=1):
                lines.append(f"  {i}. {recommendation}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def format_json(self, report: GapReport) -> str:
        """Render the report as a JSON string.

        Parameters
        ----------
        report:
            The :class:`GapReport` to format.

        Returns
        -------
        str
            Pretty-printed JSON with 2-space indentation.
        """
        payload: dict[str, object] = {
            "report_id": report.report_id,
            "created_at": report.created_at.isoformat(),
            "overall_score": report.overall_score,
            "summary": report.summary,
            "recommendations": list(report.recommendations),
            "dimensions": {
                dimension_name: [
                    {
                        "metric": gap.metric.value,
                        "value": gap.value,
                        "interpretation": gap.interpretation,
                    }
                    for gap in gaps
                ]
                for dimension_name, gaps in report.dimensions.items()
            },
        }
        return json.dumps(payload, indent=2)

    def format_markdown(self, report: GapReport) -> str:
        """Render the report as a Markdown document.

        Parameters
        ----------
        report:
            The :class:`GapReport` to format.

        Returns
        -------
        str
            Markdown-formatted report suitable for GitHub, Confluence, etc.
        """
        lines: list[str] = []
        lines.append("# Sim-to-Real Gap Report")
        lines.append("")
        lines.append(f"**Report ID:** `{report.report_id}`")
        lines.append(f"**Created:** {report.created_at.isoformat()}")
        lines.append(f"**Overall Score:** {report.overall_score:.4f} *(0 = no gap, 1 = maximum gap)*")
        lines.append("")
        lines.append(f"## Summary")
        lines.append("")
        lines.append(report.summary)
        lines.append("")

        if report.dimensions:
            lines.append("## Dimension Results")
            lines.append("")
            for dimension_name, gaps in sorted(report.dimensions.items()):
                lines.append(f"### {dimension_name}")
                lines.append("")
                lines.append("| Metric | Value | Interpretation |")
                lines.append("|--------|-------|----------------|")
                for gap in gaps:
                    lines.append(
                        f"| {gap.metric.value} | {gap.value:.6f} | {gap.interpretation} |"
                    )
                lines.append("")

        if report.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for recommendation in report.recommendations:
                lines.append(f"- {recommendation}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_summary(self, overall_score: float) -> str:
        """Produce a one-sentence summary from the overall score.

        Parameters
        ----------
        overall_score:
            Weighted average gap score in [0, 1].

        Returns
        -------
        str
            Human-readable severity summary.
        """
        if overall_score < 0.2:
            return (
                "The sim-to-real gap is low; the simulation closely matches the "
                "real-world distributions."
            )
        if overall_score < 0.5:
            return (
                "The sim-to-real gap is moderate; some dimensions show meaningful "
                "divergence that may affect transfer performance."
            )
        return (
            "The sim-to-real gap is high; significant distribution mismatches "
            "detected across multiple dimensions — transfer is likely unreliable."
        )

    def _generate_recommendations(
        self, dimension_gaps: dict[str, list[DimensionGap]]
    ) -> list[str]:
        """Produce heuristic recommendations based on gap severity.

        Rules applied (in priority order):
        - Any ``"high"`` interpretation in any dimension → recommend domain randomisation.
        - Any ``"medium"`` or ``"high"`` interpretation → recommend recalibration.
        - KL divergence is ``"high"`` → recommend re-examining reward shaping.
        - Wasserstein is ``"high"`` → recommend adjusting sim physics parameters.
        - MMD is ``"high"`` → recommend increasing sample diversity.
        - JSD is ``"high"`` → recommend reviewing observation normalisation.
        - All gaps are ``"low"`` → positive feedback.

        Parameters
        ----------
        dimension_gaps:
            Mapping from dimension name to its list of :class:`DimensionGap` values.

        Returns
        -------
        list[str]
            Deduplicated list of actionable recommendation strings.
        """
        recommendations: list[str] = []
        seen: set[str] = set()

        def _add(text: str) -> None:
            if text not in seen:
                seen.add(text)
                recommendations.append(text)

        has_high = False
        has_medium_or_high = False
        all_low = True

        high_metrics: set[GapMetric] = set()

        for gaps in dimension_gaps.values():
            for gap in gaps:
                if gap.interpretation in ("medium", "high"):
                    has_medium_or_high = True
                    all_low = False
                if gap.interpretation == "high":
                    has_high = True
                    high_metrics.add(gap.metric)

        if has_high:
            _add(
                "Apply domain randomisation to bridge high-gap dimensions before "
                "transferring policies to real hardware."
            )

        if has_medium_or_high:
            _add(
                "Re-run calibration with a larger paired dataset to reduce "
                "systematic sim-to-real bias."
            )

        if GapMetric.KL_DIVERGENCE in high_metrics:
            _add(
                "High KL divergence detected — review reward shaping and ensure "
                "the simulation policy objective aligns with real-world behaviour."
            )

        if GapMetric.WASSERSTEIN in high_metrics:
            _add(
                "High Wasserstein distance detected — consider adjusting simulation "
                "physics parameters (friction, damping, mass) to better match real dynamics."
            )

        if GapMetric.MMD in high_metrics:
            _add(
                "High MMD detected — increase the diversity of simulation rollouts "
                "to better cover the real-world distribution support."
            )

        if GapMetric.JENSEN_SHANNON in high_metrics:
            _add(
                "High Jensen-Shannon divergence detected — review observation "
                "normalisation and sensor noise models in the simulation."
            )

        if all_low:
            _add(
                "All gap metrics are low — the simulation is a good proxy for the "
                "real system; proceed with policy transfer."
            )

        return recommendations

    def __repr__(self) -> str:
        return "GapReporter()"
