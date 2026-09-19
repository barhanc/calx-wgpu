import type { WgpuExecutionContext } from './context';

/**
 * Represents a GPU compute kernel in the ExecuTorch WebGPU execution engine.
 *
 * Each kernel encapsulates:
 * 1. Its canonical PyTorch / ExecuTorch operator identifier (`name`, e.g.
 *    `'aten::add.Tensor'`).
 * 2. Its raw WebGPU Shading Language source code (`wgsl`), which can be a single
 *    shader string or a record of named shader variants
 *    (e.g. `{ tiled: string, vec4: string }`).
 * 3. Its record construction logic (`recordIn`), which validates input/output
 *    tensors, allocates shape metadata uniform buffers, compiles or retrieves
 *    cached compute pipelines, and records the resulting compute dispatch into
 *    the given {@link WgpuExecutionContext}.
 * @typeParam TArgs The tuple schema of value IDs accepted by this kernel's
 * {@link recordIn} method.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Kernel<TArgs extends readonly PropertyKey[] = any> = {
  /**
   * The canonical ExecuTorch operator target name. Matches operator node
   * schemas in the `.pte` FlatBuffer (e.g. `'aten::add.Tensor'`).
   */
  readonly name: string;

  /**
   * The complete WGSL compute shader source code or variant record for this kernel.
   * Exposes compile-time pipeline override constants (e.g. workgroup sizes,
   * scalars).
   */
  readonly wgsl: string | Record<string, string>;

  /**
   * Resolves tensor and scalar arguments from the execution context, validates
   * their dtypes, ranks, and broadcast shapes, constructs uniform metadata
   * buffers, and records the compute pass into the execution context.
   * @param ctx The {@link WgpuExecutionContext} where tensors are stored and
   * dispatches recorded.
   * @param args Arguments referring to entries in `ctx` (inputs, scalars,
   * output).
   * @throws {Error} If arguments are invalid, dtypes mismatch, or shapes are
   * not broadcastable.
   */
  readonly recordIn: (ctx: WgpuExecutionContext, args: TArgs) => void;
};
