/**
 * @aumos/agent-sim-bridge
 *
 * TypeScript client for the AumOS agent-sim-bridge sim-to-real transfer layer.
 * Provides HTTP client, simulation environment management, domain randomization,
 * gap estimation, and reality adapter type definitions.
 */

// Client and configuration
export type { AgentSimBridgeClient, AgentSimBridgeClientConfig } from "./client.js";
export { createAgentSimBridgeClient } from "./client.js";

// Core types
export type {
  ApiError,
  ApiResult,
  DistributionType,
  DomainRandomization,
  EnvironmentInfo,
  GapDimension,
  GapEstimation,
  GapEstimationRequest,
  GapEstimationResult,
  GapMetric,
  RandomizationRequest,
  RandomizationResult,
  SimResult,
  SimulationConfig,
  SimulationEnvironment,
  SpaceSpec,
  TransferConfig,
  TransferResult,
} from "./types.js";
