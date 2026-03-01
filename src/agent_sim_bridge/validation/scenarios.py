"""Standard validation scenarios for sim-to-real transfer testing.

Each ValidationScenario bundles a set of inputs with the expected
properties that a real environment should exhibit.  The five standard
categories cover the most common text-based agent tasks:

* classification  — label assignment
* extraction      — structured data extraction from text
* qa              — open-domain question answering
* summarization   — length-reducing restatement
* multi_step      — chained reasoning across multiple turns
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent_sim_bridge.validation.environment import EnvironmentInput


@dataclass(frozen=True)
class ValidationScenario:
    """A self-contained validation scenario.

    Attributes
    ----------
    id:
        Unique identifier for this scenario.
    name:
        Short human-readable label.
    description:
        Longer explanation of what the scenario tests.
    category:
        One of ``classification``, ``extraction``, ``qa``,
        ``summarization``, or ``multi_step``.
    inputs:
        Ordered list of inputs to send to each environment.
    expected_properties:
        Specification of properties that correct outputs should satisfy.
        The harness records these in the ScenarioResult for downstream
        analysis; it does not enforce them automatically.
    """

    id: str
    name: str
    description: str
    category: str
    inputs: list[EnvironmentInput]
    expected_properties: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Standard Scenarios
# ---------------------------------------------------------------------------

STANDARD_SCENARIOS: list[ValidationScenario] = [
    # ------------------------------------------------------------------
    # 1. Text Classification
    # ------------------------------------------------------------------
    ValidationScenario(
        id="scenario_classification",
        name="Text Classification",
        description=(
            "Classify the sentiment of product reviews. "
            "Expected labels: positive, negative, neutral."
        ),
        category="classification",
        inputs=[
            EnvironmentInput(
                prompt="Classify sentiment: 'Great product, works perfectly!'",
                scenario_id="scenario_classification",
            ),
            EnvironmentInput(
                prompt="Classify sentiment: 'Terrible quality, broke immediately.'",
                scenario_id="scenario_classification",
            ),
            EnvironmentInput(
                prompt="Classify sentiment: 'It works as expected, nothing special.'",
                scenario_id="scenario_classification",
            ),
        ],
        expected_properties={
            "output_contains_label": True,
            "labels": ["positive", "negative", "neutral"],
        },
    ),
    # ------------------------------------------------------------------
    # 2. Named Entity Extraction
    # ------------------------------------------------------------------
    ValidationScenario(
        id="scenario_extraction",
        name="Named Entity Extraction",
        description=(
            "Extract named entities (person, organisation, location) "
            "from short paragraphs."
        ),
        category="extraction",
        inputs=[
            EnvironmentInput(
                prompt=(
                    "Extract entities: "
                    "'Alice Smith joined Acme Corp in New York last Monday.'"
                ),
                scenario_id="scenario_extraction",
            ),
            EnvironmentInput(
                prompt=(
                    "Extract entities: "
                    "'Dr. Lee presented at the MIT conference in Boston.'"
                ),
                scenario_id="scenario_extraction",
            ),
            EnvironmentInput(
                prompt=(
                    "Extract entities: "
                    "'The CEO of TechStart, Bob Jones, visited London.'"
                ),
                scenario_id="scenario_extraction",
            ),
        ],
        expected_properties={
            "entity_types": ["PERSON", "ORG", "LOCATION"],
            "structured_output": True,
        },
    ),
    # ------------------------------------------------------------------
    # 3. Open-Domain Question Answering
    # ------------------------------------------------------------------
    ValidationScenario(
        id="scenario_qa",
        name="Question Answering",
        description=(
            "Answer factual questions with a concise, direct response."
        ),
        category="qa",
        inputs=[
            EnvironmentInput(
                prompt="Answer the question: 'What is the capital of France?'",
                scenario_id="scenario_qa",
            ),
            EnvironmentInput(
                prompt=(
                    "Answer the question: "
                    "'How many days are there in a leap year?'"
                ),
                scenario_id="scenario_qa",
            ),
            EnvironmentInput(
                prompt=(
                    "Answer the question: "
                    "'What programming language was Python named after?'"
                ),
                scenario_id="scenario_qa",
            ),
        ],
        expected_properties={
            "concise_answer": True,
            "max_tokens": 50,
        },
    ),
    # ------------------------------------------------------------------
    # 4. Summarization
    # ------------------------------------------------------------------
    ValidationScenario(
        id="scenario_summarization",
        name="Text Summarization",
        description=(
            "Produce a one-sentence summary shorter than the original input."
        ),
        category="summarization",
        inputs=[
            EnvironmentInput(
                prompt=(
                    "Summarize in one sentence: "
                    "'The transformer architecture, introduced in 2017, "
                    "replaced recurrent networks with self-attention mechanisms. "
                    "It enabled parallel processing and became the foundation "
                    "for large language models.'"
                ),
                scenario_id="scenario_summarization",
            ),
            EnvironmentInput(
                prompt=(
                    "Summarize in one sentence: "
                    "'Reinforcement learning trains agents through trial and error. "
                    "An agent interacts with an environment, receives rewards or "
                    "penalties, and updates its policy to maximise cumulative reward.'"
                ),
                scenario_id="scenario_summarization",
            ),
            EnvironmentInput(
                prompt=(
                    "Summarize in one sentence: "
                    "'Docker containers package application code together with its "
                    "dependencies into a portable unit. They run identically across "
                    "development, staging, and production environments.'"
                ),
                scenario_id="scenario_summarization",
            ),
        ],
        expected_properties={
            "output_shorter_than_input": True,
            "single_sentence": True,
        },
    ),
    # ------------------------------------------------------------------
    # 5. Multi-Step Reasoning
    # ------------------------------------------------------------------
    ValidationScenario(
        id="scenario_multi_step",
        name="Multi-Step Reasoning",
        description=(
            "Solve problems that require sequential logical steps "
            "before arriving at an answer."
        ),
        category="multi_step",
        inputs=[
            EnvironmentInput(
                prompt=(
                    "Solve step by step: "
                    "'A store sells apples for $0.50 each. "
                    "Alice buys 6 apples and pays with a $5 bill. "
                    "How much change does she receive?'"
                ),
                scenario_id="scenario_multi_step",
            ),
            EnvironmentInput(
                prompt=(
                    "Solve step by step: "
                    "'A train travels at 60 km/h for 2 hours, "
                    "then at 80 km/h for 1.5 hours. "
                    "What is the total distance covered?'"
                ),
                scenario_id="scenario_multi_step",
            ),
            EnvironmentInput(
                prompt=(
                    "Solve step by step: "
                    "'If today is Wednesday and a meeting is in 10 days, "
                    "on what day of the week does the meeting fall?'"
                ),
                scenario_id="scenario_multi_step",
            ),
        ],
        expected_properties={
            "shows_reasoning_steps": True,
            "numeric_answer": True,
        },
    ),
]
