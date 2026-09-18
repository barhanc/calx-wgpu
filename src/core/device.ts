/**
 * Configuration options for initializing a WebGPU device.
 */
export type DeviceOptions = {
  /** Request a high-performance or low-power GPU (default: 'high-performance'). */
  readonly powerPreference?: GPUPowerPreference;
  /** Whether to enable 16-bit floating point math if supported by GPU hardware (default: true). */
  readonly enableFloat16?: boolean;
  /** Whether to enable subgroup operations if supported by GPU hardware (default: true). */
  readonly enableSubgroups?: boolean;
  /** Additional required WebGPU features to request from the adapter. */
  readonly requiredFeatures?: GPUFeatureName[];
  /** Required WebGPU limits to request from the adapter. */
  readonly requiredLimits?: Record<string, number>;
};

/**
 * Synchronously checks whether the current execution environment supports
 * WebGPU.
 * @returns True if WebGPU is available in the current environment.
 */
export function isWebGPUSupported(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator && Boolean(navigator.gpu);
}

/**
 * Initializes and configures a WebGPU device.
 *
 * Checks for WebGPU support, requests a suitable adapter with the requested
 * power preference, enables 16-bit float (`shader-f16`) and subgroup operations
 * (`subgroups`) when available, and requests the device.
 *
 * @param options Optional configuration parameters for adapter and device
 * initialization.
 *
 * @returns A promise that resolves to the initialized WebGPU device.
 */
export async function initDevice(options: DeviceOptions = {}): Promise<GPUDevice> {
  if (!isWebGPUSupported()) {
    throw new Error('WebGPU is not supported in this environment (navigator.gpu is missing)');
  }

  const {
    powerPreference = 'high-performance',
    enableFloat16 = true,
    enableSubgroups = true,
    requiredFeatures = [],
    requiredLimits,
  } = options;

  const adapter = await navigator.gpu.requestAdapter({ powerPreference });
  if (!adapter) {
    throw new Error('Failed to obtain a WebGPU adapter. No suitable GPU device was found.');
  }

  const featuresToRequest = new Set<GPUFeatureName>(requiredFeatures);

  if (enableFloat16 && adapter.features.has('shader-f16')) {
    featuresToRequest.add('shader-f16');
  }
  if (enableSubgroups && adapter.features.has('subgroups')) {
    featuresToRequest.add('subgroups');
  }

  return await adapter.requestDevice({
    requiredFeatures: Array.from(featuresToRequest),
    requiredLimits,
  });
}
