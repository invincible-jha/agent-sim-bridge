/**
 * HTTP client for the agent-sim-bridge sim-to-real transfer API.
 *
 * Delegates all HTTP transport to `@aumos/sdk-core` which provides
 * automatic retry with exponential back-off, timeout management via
 * `AbortSignal.timeout`, interceptor support, and a typed error hierarchy.
 *
 * The public-facing `ApiResult<T>` envelope is preserved for full
 * backward compatibility with existing callers.
 *
 * @example
 * ```ts
 * import { createAgentSimBridgeClient } from "@aumos/agent-sim-bridge";
 *
 * const client = createAgentSimBridgeClient({ baseUrl: "http://localhost:8094" });
 *
 * const sim = await client.createSimulation({
 *   backend: "pybullet",
 *   name: "manipulation-v1",
 *   max_episode_steps: 500,
 * });
 *
 * if (sim.ok) {
 *   console.log("Simulation created:", sim.data.environment_id);
 * }
 * ```
 */

import {
  createHttpClient,
  HttpError,
  NetworkError,
  TimeoutError,
  AumosError,
  type HttpClient,
} from "@aumos/sdk-core";

import type {
  ApiResult,
  DomainRandomization,
  GapEstimationRequest,
  GapEstimationResult,
  RandomizationRequest,
  RandomizationResult,
  SimResult,
  SimulationConfig,
  SimulationEnvironment,
  TransferConfig,
  TransferResult,
} from "./types.js";

// ---------------------------------------------------------------------------
// Client configuration
// ---------------------------------------------------------------------------

/** Configuration options for the AgentSimBridgeClient. */
export interface AgentSimBridgeClientConfig {
  /** Base URL of the agent-sim-bridge server (e.g. "http://localhost:8094"). */
  readonly baseUrl: string;
  /** Optional request timeout in milliseconds (default: 30000). */
  readonly timeoutMs?: number;
  /** Optional extra HTTP headers sent with every request. */
  readonly headers?: Readonly<Record<string, string>>;
}

// ---------------------------------------------------------------------------
// Internal adapter
// ---------------------------------------------------------------------------

async function callApi<T>(
  operation: () => Promise<{ readonly data: T; readonly status: number }>,
): Promise<ApiResult<T>> {
  try {
    const response = await operation();
    return { ok: true, data: response.data };
  } catch (error: unknown) {
    if (error instanceof HttpError) {
      return {
        ok: false,
        error: { error: error.message, detail: String(error.body ?? "") },
        status: error.statusCode,
      };
    }
    if (error instanceof TimeoutError) {
      return {
        ok: false,
        error: { error: "Request timed out", detail: error.message },
        status: 0,
      };
    }
    if (error instanceof NetworkError) {
      return {
        ok: false,
        error: { error: "Network error", detail: error.message },
        status: 0,
      };
    }
    if (error instanceof AumosError) {
      return {
        ok: false,
        error: { error: error.code, detail: error.message },
        status: error.statusCode ?? 0,
      };
    }
    const message = error instanceof Error ? error.message : String(error);
    return {
      ok: false,
      error: { error: "Unexpected error", detail: message },
      status: 0,
    };
  }
}

// ---------------------------------------------------------------------------
// Client interface
// ---------------------------------------------------------------------------

/** Typed HTTP client for the agent-sim-bridge server. */
export interface AgentSimBridgeClient {
  /**
   * Create and register a new simulation environment.
   *
   * @param config - Simulation configuration including backend and episode settings.
   * @returns The created SimulationEnvironment descriptor.
   */
  createSimulation(
    config: SimulationConfig,
  ): Promise<ApiResult<SimulationEnvironment>>;

  /**
   * Transfer simulation values to real-world units using a calibration profile.
   *
   * @param options - Transfer configuration and simulation values to transform.
   * @returns A TransferResult with transformed real-world values.
   */
  transferToReality(options: {
    config: TransferConfig;
    sim_values: readonly number[];
  }): Promise<ApiResult<TransferResult>>;

  /**
   * Estimate the sim-to-real distribution gap across multiple dimensions.
   *
   * @param request - Gap estimation request with dimensions and metric selection.
   * @returns A GapEstimationResult with per-dimension gaps and overall score.
   */
  estimateGap(
    request: GapEstimationRequest,
  ): Promise<ApiResult<GapEstimationResult>>;

  /**
   * Get the results for a completed simulation episode.
   *
   * @param options - Environment ID and optional run ID filter.
   * @returns Array of SimResult records ordered by completion time descending.
   */
  getSimResults(options: {
    environment_id: string;
    run_id?: string;
    limit?: number;
  }): Promise<ApiResult<readonly SimResult[]>>;

  /**
   * Configure and apply domain randomization to a simulation environment.
   *
   * @param request - Randomization request with environment ID and parameter specs.
   * @returns A RandomizationResult with the sampled parameter values.
   */
  configureRandomization(
    request: RandomizationRequest,
  ): Promise<ApiResult<RandomizationResult>>;

  /**
   * List all registered simulation environments.
   *
   * @param options - Optional filter parameters.
   * @returns Array of SimulationEnvironment descriptors.
   */
  listSimulations(options?: {
    backend?: string;
    limit?: number;
  }): Promise<ApiResult<readonly SimulationEnvironment[]>>;

  /**
   * Get the transfer configuration (calibration profile) for an environment.
   *
   * @param environmentId - The simulation environment identifier.
   * @returns The TransferConfig describing the linear sim-to-real transform.
   */
  getTransferConfig(
    environmentId: string,
  ): Promise<ApiResult<TransferConfig>>;

  /**
   * Update the transfer configuration for a simulation environment.
   *
   * @param environmentId - The simulation environment identifier.
   * @param config - The new TransferConfig to apply.
   * @returns The updated TransferConfig.
   */
  updateTransferConfig(
    environmentId: string,
    config: TransferConfig,
  ): Promise<ApiResult<TransferConfig>>;

  /**
   * List the randomization configurations for a simulation environment.
   *
   * @param environmentId - The simulation environment identifier.
   * @returns Array of DomainRandomization specifications.
   */
  listRandomizations(
    environmentId: string,
  ): Promise<ApiResult<readonly DomainRandomization[]>>;
}

// ---------------------------------------------------------------------------
// Client factory
// ---------------------------------------------------------------------------

/**
 * Create a typed HTTP client for the agent-sim-bridge server.
 *
 * @param config - Client configuration including base URL.
 * @returns An AgentSimBridgeClient instance.
 */
export function createAgentSimBridgeClient(
  config: AgentSimBridgeClientConfig,
): AgentSimBridgeClient {
  const http: HttpClient = createHttpClient({
    baseUrl: config.baseUrl,
    timeout: config.timeoutMs ?? 30_000,
    defaultHeaders: config.headers,
  });

  return {
    createSimulation(
      simConfig: SimulationConfig,
    ): Promise<ApiResult<SimulationEnvironment>> {
      return callApi(() =>
        http.post<SimulationEnvironment>("/sim-bridge/simulations", simConfig),
      );
    },

    transferToReality(options: {
      config: TransferConfig;
      sim_values: readonly number[];
    }): Promise<ApiResult<TransferResult>> {
      return callApi(() =>
        http.post<TransferResult>("/sim-bridge/transfer/sim-to-real", options),
      );
    },

    estimateGap(request: GapEstimationRequest): Promise<ApiResult<GapEstimationResult>> {
      return callApi(() =>
        http.post<GapEstimationResult>("/sim-bridge/gap/estimate", request),
      );
    },

    getSimResults(options: {
      environment_id: string;
      run_id?: string;
      limit?: number;
    }): Promise<ApiResult<readonly SimResult[]>> {
      const queryParams: Record<string, string> = {
        environment_id: options.environment_id,
      };
      if (options.run_id !== undefined) queryParams["run_id"] = options.run_id;
      if (options.limit !== undefined) queryParams["limit"] = String(options.limit);
      return callApi(() =>
        http.get<readonly SimResult[]>("/sim-bridge/simulations/results", { queryParams }),
      );
    },

    configureRandomization(
      request: RandomizationRequest,
    ): Promise<ApiResult<RandomizationResult>> {
      return callApi(() =>
        http.post<RandomizationResult>("/sim-bridge/randomization/apply", request),
      );
    },

    listSimulations(options?: {
      backend?: string;
      limit?: number;
    }): Promise<ApiResult<readonly SimulationEnvironment[]>> {
      const queryParams: Record<string, string> = {};
      if (options?.backend !== undefined) queryParams["backend"] = options.backend;
      if (options?.limit !== undefined) queryParams["limit"] = String(options.limit);
      return callApi(() =>
        http.get<readonly SimulationEnvironment[]>("/sim-bridge/simulations", {
          queryParams,
        }),
      );
    },

    getTransferConfig(environmentId: string): Promise<ApiResult<TransferConfig>> {
      return callApi(() =>
        http.get<TransferConfig>(
          `/sim-bridge/simulations/${encodeURIComponent(environmentId)}/transfer-config`,
        ),
      );
    },

    updateTransferConfig(
      environmentId: string,
      transferConfig: TransferConfig,
    ): Promise<ApiResult<TransferConfig>> {
      return callApi(() =>
        http.put<TransferConfig>(
          `/sim-bridge/simulations/${encodeURIComponent(environmentId)}/transfer-config`,
          transferConfig,
        ),
      );
    },

    listRandomizations(
      environmentId: string,
    ): Promise<ApiResult<readonly DomainRandomization[]>> {
      return callApi(() =>
        http.get<readonly DomainRandomization[]>(
          `/sim-bridge/simulations/${encodeURIComponent(environmentId)}/randomizations`,
        ),
      );
    },
  };
}
