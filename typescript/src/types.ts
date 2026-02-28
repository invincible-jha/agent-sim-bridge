/**
 * TypeScript interfaces for the agent-sim-bridge sim-to-real transfer layer.
 *
 * Mirrors the Python dataclasses and Pydantic models defined in:
 *   agent_sim_bridge.environment.base          (SpaceSpec, EnvironmentInfo)
 *   agent_sim_bridge.environment.sim_env       (SimulationEnvironment)
 *   agent_sim_bridge.transfer.bridge           (CalibrationProfile)
 *   agent_sim_bridge.transfer.domain_randomization (DistributionType, RandomizationConfig)
 *   agent_sim_bridge.gap.estimator             (GapMetric, GapDimension, DimensionGap)
 *   agent_sim_bridge.staging.results           (SimResult)
 *
 * All interfaces use readonly fields to match Python frozen dataclasses.
 */

// ---------------------------------------------------------------------------
// Space and environment types
// ---------------------------------------------------------------------------

/**
 * Describes a continuous box space (observation or action).
 * Maps to SpaceSpec Pydantic model in Python.
 */
export interface SpaceSpec {
  /**
   * Dimensionality tuple (e.g. [3] for a 3-vector, [84, 84, 3] for an image).
   * Represented as an array in TypeScript.
   */
  readonly shape: readonly number[];
  /** Per-dimension lower bounds. null means negative infinity. */
  readonly low: readonly number[] | null;
  /** Per-dimension upper bounds. null means positive infinity. */
  readonly high: readonly number[] | null;
  /** Numpy dtype string, e.g. "float32". */
  readonly dtype: string;
}

/** Metadata describing an environment instance. */
export interface EnvironmentInfo {
  /** Human-readable identifier. */
  readonly name: string;
  /** Semantic version string. */
  readonly version: string;
  /** True for simulation environments, false for real-world. */
  readonly is_simulation: boolean;
  /** Maximum allowed steps per episode before truncation. */
  readonly max_episode_steps: number;
  /** Arbitrary key/value metadata. */
  readonly metadata: Readonly<Record<string, unknown>>;
}

// ---------------------------------------------------------------------------
// Simulation environment
// ---------------------------------------------------------------------------

/**
 * Describes a simulation environment configuration.
 * Corresponds to the SimulationEnvironment class in Python.
 */
export interface SimulationEnvironment {
  /** Unique identifier for this simulation environment. */
  readonly environment_id: string;
  /** Metadata about this environment instance. */
  readonly info: EnvironmentInfo;
  /** Description of the observation (state) space. */
  readonly state_space: SpaceSpec;
  /** Description of the action space. */
  readonly action_space: SpaceSpec;
  /** Whether trajectory recording is enabled. */
  readonly record_trajectories: boolean;
  /** ISO-8601 UTC timestamp when this environment was created. */
  readonly created_at: string;
}

/** Configuration for creating a simulation environment. */
export interface SimulationConfig {
  /** The backend to use (e.g. "pybullet", "gazebo", "generic"). */
  readonly backend: string;
  /** Human-readable name for this environment instance. */
  readonly name?: string;
  /** Episode truncation limit. */
  readonly max_episode_steps?: number;
  /** Whether to enable trajectory recording. */
  readonly record_trajectories?: boolean;
  /** Backend-specific configuration parameters. */
  readonly backend_config?: Readonly<Record<string, unknown>>;
}

// ---------------------------------------------------------------------------
// Sim result
// ---------------------------------------------------------------------------

/** The result of a simulation episode or step. */
export interface SimResult {
  /** Unique identifier for this simulation run. */
  readonly run_id: string;
  /** The environment that produced this result. */
  readonly environment_id: string;
  /** Total number of steps completed in this episode. */
  readonly total_steps: number;
  /** Cumulative reward over the episode. */
  readonly total_reward: number;
  /** Whether the episode ended due to a terminal state. */
  readonly terminated: boolean;
  /** Whether the episode ended due to truncation. */
  readonly truncated: boolean;
  /** ISO-8601 UTC timestamp when the episode ended. */
  readonly completed_at: string;
  /** Arbitrary diagnostic information. */
  readonly info: Readonly<Record<string, unknown>>;
}

// ---------------------------------------------------------------------------
// Transfer / calibration types
// ---------------------------------------------------------------------------

/**
 * Encodes a linear sim-to-real calibration transform.
 * Maps to CalibrationProfile dataclass in Python.
 *
 * Each dimension i transforms as: real[i] = sim[i] * scale_factors[i] + offsets[i]
 */
export interface TransferConfig {
  /** Per-dimension multiplicative scale factors. */
  readonly scale_factors: readonly number[];
  /** Per-dimension additive offsets (applied after scaling). */
  readonly offsets: readonly number[];
  /** Optional name of the noise model to apply after transformation. */
  readonly noise_model: string | null;
  /** Optional human-readable labels for each dimension. */
  readonly dimension_names: readonly string[];
  /** Arbitrary annotations (e.g. calibration date, sensor serial numbers). */
  readonly metadata: Readonly<Record<string, unknown>>;
}

/** Result of a sim-to-real transfer operation. */
export interface TransferResult {
  /** The configuration used for the transfer. */
  readonly config: TransferConfig;
  /** Input values (simulation units). */
  readonly sim_values: readonly number[];
  /** Output values (real-world units). */
  readonly real_values: readonly number[];
  /** ISO-8601 UTC timestamp of the transfer. */
  readonly transferred_at: string;
}

// ---------------------------------------------------------------------------
// Domain randomization types
// ---------------------------------------------------------------------------

/**
 * Supported sampling distributions for domain randomization.
 * Maps to DistributionType enum in Python.
 */
export type DistributionType = "uniform" | "gaussian" | "log_uniform" | "constant";

/**
 * Specification for randomizing one environment parameter.
 * Maps to RandomizationConfig dataclass in Python.
 */
export interface DomainRandomization {
  /**
   * Dot-path name of the parameter to randomize.
   * e.g. "physics.gravity" or "sensor.noise_std".
   */
  readonly parameter_name: string;
  /** Sampling distribution to use. */
  readonly distribution: DistributionType;
  /** Lower bound (or mean for GAUSSIAN, or constant value for CONSTANT). */
  readonly low: number;
  /** Upper bound (or std dev for GAUSSIAN). */
  readonly high: number;
  /** Optional hard minimum applied after sampling. */
  readonly clip_low: number | null;
  /** Optional hard maximum applied after sampling. */
  readonly clip_high: number | null;
  /** Arbitrary annotations. */
  readonly metadata: Readonly<Record<string, unknown>>;
}

/** Configuration for applying a set of domain randomizations. */
export interface RandomizationRequest {
  /** The environment identifier to apply randomizations to. */
  readonly environment_id: string;
  /** The randomization specifications to apply. */
  readonly randomizations: readonly DomainRandomization[];
  /** Optional RNG seed for reproducibility. */
  readonly seed?: number;
}

/** Result of applying domain randomizations to an environment. */
export interface RandomizationResult {
  /** The environment that was randomized. */
  readonly environment_id: string;
  /** The sampled parameter values (parameter_name -> sampled_value). */
  readonly sampled_parameters: Readonly<Record<string, number>>;
  /** ISO-8601 UTC timestamp of when randomization was applied. */
  readonly applied_at: string;
}

// ---------------------------------------------------------------------------
// Gap estimation types
// ---------------------------------------------------------------------------

/**
 * Statistical distance metric for comparing distributions.
 * Maps to GapMetric enum in Python.
 */
export type GapMetric =
  | "KL_DIVERGENCE"
  | "WASSERSTEIN"
  | "MMD"
  | "JENSEN_SHANNON";

/**
 * A named pair of sim and real distributions for one observable dimension.
 * Maps to GapDimension frozen dataclass in Python.
 */
export interface GapDimension {
  /** Human-readable label (e.g. "joint_velocity_0"). */
  readonly name: string;
  /** Empirical distribution collected in simulation. */
  readonly sim_distribution: readonly number[];
  /** Empirical distribution collected on the real system. */
  readonly real_distribution: readonly number[];
}

/**
 * The gap value for one metric applied to one dimension.
 * Maps to DimensionGap frozen dataclass in Python.
 */
export interface GapEstimation {
  /** The name of the GapDimension this result belongs to. */
  readonly dimension_name: string;
  /** The GapMetric used to compute the value. */
  readonly metric: GapMetric;
  /** The computed distance value. */
  readonly value: number;
  /** Qualitative severity: "low", "medium", or "high". */
  readonly interpretation: string;
}

/** Request for estimating the sim-to-real gap across dimensions. */
export interface GapEstimationRequest {
  /** Dimensions to analyse. */
  readonly dimensions: readonly GapDimension[];
  /** Metrics to compute (defaults to all four if omitted). */
  readonly metrics?: readonly GapMetric[];
}

/** Full result of a sim-to-real gap estimation run. */
export interface GapEstimationResult {
  /** Per-dimension gap results, indexed by dimension name. */
  readonly dimension_gaps: Readonly<Record<string, readonly GapEstimation[]>>;
  /** Weighted average gap score in [0, 1]. */
  readonly overall_gap_score: number;
  /** ISO-8601 UTC timestamp when the estimation was computed. */
  readonly estimated_at: string;
}

// ---------------------------------------------------------------------------
// API result wrapper (shared pattern)
// ---------------------------------------------------------------------------

/** Standard error payload returned by the agent-sim-bridge API. */
export interface ApiError {
  readonly error: string;
  readonly detail: string;
}

/** Result type for all client operations. */
export type ApiResult<T> =
  | { readonly ok: true; readonly data: T }
  | { readonly ok: false; readonly error: ApiError; readonly status: number };
