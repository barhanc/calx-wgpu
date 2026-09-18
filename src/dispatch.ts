/**
 * Binding specification for constructing bind groups.
 */
export type BindingSpec = {
  /** The shader binding index corresponding to `@binding(n)` in WGSL. */
  readonly binding: number;
  /** The underlying WebGPU buffer to bind. */
  readonly buffer: GPUBuffer;
  /** Optional byte offset from the start of the buffer (defaults to 0). */
  readonly offset?: number;
  /** Optional size in bytes of the buffer slice accessible to the shader. */
  readonly size?: number;
};

/**
 * Result of creating a compute pipeline and matching bind group.
 */
export type ComputePipelineBundle = {
  /** The compiled or cached WebGPU compute pipeline. */
  readonly pipeline: GPUComputePipeline;
  /** The bind group pre-configured to match the pipeline's layout. */
  readonly bindGroup: GPUBindGroup;
};

/**
 * A recorded GPU compute dispatch command.
 */
export type WgpuDispatch = {
  /** The compute pipeline to bind for execution. */
  readonly pipeline: GPUComputePipeline;
  /** The bind group containing bound buffers and uniforms. */
  readonly bindGroup: GPUBindGroup;
  /** Number of workgroups to dispatch in the X dimension. */
  readonly workgroupCountX: number;
  /** Number of workgroups to dispatch in the Y dimension. */
  readonly workgroupCountY: number;
  /** Optional number of workgroups to dispatch in the Z dimension (defaults to 1). */
  readonly workgroupCountZ?: number;
};

const pipelineCache = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();

/**
 * Serializes pipeline override constants into a deterministic,
 * order-independent cache key string.
 * @param constants Pipeline override constants.
 * @returns Serialized key string.
 */
function serializeConstants(constants?: Record<string, number>): string {
  if (!constants) return '';
  const entries = Object.entries(constants);
  if (entries.length === 0) return '';
  entries.sort(([a], [b]) => a.localeCompare(b));
  return entries.map(([k, v]) => `${k}=${v}`).join(';');
}

/**
 * Builds and caches a compute pipeline along with its bind group from a list of
 * bindings.
 *
 * @param device WebGPU device instance.
 * @param shader WGSL shader code string.
 * @param bindings List of buffer bindings for @group(0).
 * @param constants Optional pipeline override constants.
 * @param entryPoint Shader entry point function name (default: 'main').
 * @returns The compiled pipeline and its matching bind group.
 */
export function createComputeBundle(
  device: GPUDevice,
  shader: string,
  bindings: readonly BindingSpec[],
  constants: Record<string, number> = {},
  entryPoint: string = 'main'
): ComputePipelineBundle {
  let deviceMap = pipelineCache.get(device);
  if (!deviceMap) {
    deviceMap = new Map();
    pipelineCache.set(device, deviceMap);
  }

  const serializedConsts = serializeConstants(constants);
  const cacheKey = `${entryPoint}#${serializedConsts}#${shader}`;

  let pipeline = deviceMap.get(cacheKey);
  if (!pipeline) {
    const module = device.createShaderModule({ code: shader });
    const computeStage: GPUProgrammableStage = { module, entryPoint, constants };
    pipeline = device.createComputePipeline({ layout: 'auto', compute: computeStage });
    deviceMap.set(cacheKey, pipeline);
  }

  const bindGroupEntries: GPUBindGroupEntry[] = bindings.map((b) => ({
    binding: b.binding,
    resource: { buffer: b.buffer, offset: b.offset, size: b.size },
  }));

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bindGroupEntries,
  });

  return { pipeline, bindGroup };
}
