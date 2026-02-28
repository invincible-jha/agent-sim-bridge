/**
 * HTTP client for the agent-sim-bridge sim-to-real transfer API.
 *
 * Uses the Fetch API (available natively in Node 18+, browsers, and Deno).
 * No external dependencies required.
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

import type {
  ApiError,
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
// Internal helpers
// ---------------------------------------------------------------------------

async function fetchJson<T>(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<ApiResult<T>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    clearTimeout(timeoutId);

    const body = await response.json() as unknown;

    if (!response.ok) {
      const errorBody = body as Partial<ApiError>;
      return {
        ok: false,
        error: {
          error: errorBody.error ?? "Unknown error",
          detail: errorBody.detail ?? "",
        },
        status: response.status,
      };
    }

    return { ok: true, data: body as T };
  } catch (err: unknown) {
    clearTimeout(timeoutId);
    const message = err instanceof Error ? err.message : String(err);
    return {
      ok: false,
      error: { error: "Network error", detail: message },
      status: 0,
    };
  }
}

function buildHeaders(
  extraHeaders: Readonly<Record<string, string>> | undefined,
): Record<string, string> {
  return {
    "Content-Type": "application/json",
    Accept: "application/json",
    ...extraHeaders,
  };
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
  const { baseUrl, timeoutMs = 30_000, headers: extraHeaders } = config;
  const baseHeaders = buildHeaders(extraHeaders);

  return {
    async createSimulation(
      simConfig: SimulationConfig,
    ): Promise<ApiResult<SimulationEnvironment>> {
      return fetchJson<SimulationEnvironment>(
        `${baseUrl}/sim-bridge/simulations`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(simConfig),
        },
        timeoutMs,
      );
    },

    async transferToReality(options: {
      config: TransferConfig;
      sim_values: readonly number[];
    }): Promise<ApiResult<TransferResult>> {
      return fetchJson<TransferResult>(
        `${baseUrl}/sim-bridge/transfer/sim-to-real`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(options),
        },
        timeoutMs,
      );
    },

    async estimateGap(
      request: GapEstimationRequest,
    ): Promise<ApiResult<GapEstimationResult>> {
      return fetchJson<GapEstimationResult>(
        `${baseUrl}/sim-bridge/gap/estimate`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(request),
        },
        timeoutMs,
      );
    },

    async getSimResults(options: {
      environment_id: string;
      run_id?: string;
      limit?: number;
    }): Promise<ApiResult<readonly SimResult[]>> {
      const params = new URLSearchParams();
      params.set("environment_id", options.environment_id);
      if (options.run_id !== undefined) {
        params.set("run_id", options.run_id);
      }
      if (options.limit !== undefined) {
        params.set("limit", String(options.limit));
      }
      return fetchJson<readonly SimResult[]>(
        `${baseUrl}/sim-bridge/simulations/results?${params.toString()}`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async configureRandomization(
      request: RandomizationRequest,
    ): Promise<ApiResult<RandomizationResult>> {
      return fetchJson<RandomizationResult>(
        `${baseUrl}/sim-bridge/randomization/apply`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(request),
        },
        timeoutMs,
      );
    },

    async listSimulations(options?: {
      backend?: string;
      limit?: number;
    }): Promise<ApiResult<readonly SimulationEnvironment[]>> {
      const params = new URLSearchParams();
      if (options?.backend !== undefined) {
        params.set("backend", options.backend);
      }
      if (options?.limit !== undefined) {
        params.set("limit", String(options.limit));
      }
      const query = params.toString();
      return fetchJson<readonly SimulationEnvironment[]>(
        `${baseUrl}/sim-bridge/simulations${query ? `?${query}` : ""}`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async getTransferConfig(
      environmentId: string,
    ): Promise<ApiResult<TransferConfig>> {
      return fetchJson<TransferConfig>(
        `${baseUrl}/sim-bridge/simulations/${encodeURIComponent(environmentId)}/transfer-config`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async updateTransferConfig(
      environmentId: string,
      transferConfig: TransferConfig,
    ): Promise<ApiResult<TransferConfig>> {
      return fetchJson<TransferConfig>(
        `${baseUrl}/sim-bridge/simulations/${encodeURIComponent(environmentId)}/transfer-config`,
        {
          method: "PUT",
          headers: baseHeaders,
          body: JSON.stringify(transferConfig),
        },
        timeoutMs,
      );
    },

    async listRandomizations(
      environmentId: string,
    ): Promise<ApiResult<readonly DomainRandomization[]>> {
      return fetchJson<readonly DomainRandomization[]>(
        `${baseUrl}/sim-bridge/simulations/${encodeURIComponent(environmentId)}/randomizations`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },
  };
}

/** Re-export types for convenience. */
export type {
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
};
