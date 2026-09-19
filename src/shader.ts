import type { WgpuExecutionContext } from './context';

/**
 * Represents a GPU compute shader in the ExecuTorch WebGPU execution engine.
 *
 * @typeParam TArgs The tuple schema of value IDs accepted by this shader's {@link recordIn} method.
 * @typeParam TName Literal string type of the operator name.
 */
export type Shader<
  TArgs extends readonly PropertyKey[] = readonly PropertyKey[],
  TName extends string = string,
> = {
  /**
   * Canonical PyTorch / ExecuTorch operator identifier (e.g. `'aten.add.Tensor'`).
   */
  readonly name: TName;

  /**
   * Raw WebGPU Shading Language source code, either as a single shader string
   * or a dictionary of named shader variants (e.g. `{ tiled: string, vec4: string }`).
   */
  readonly code: string | Record<string, string>;

  /**
   * Dispatches and records the operator's compute pass into the execution context.
   * Validates input/output tensors, allocates uniform buffers, compiles or retrieves
   * cached pipelines, and records the dispatch.
   *
   * @param ctx The execution context to record dispatches into.
   * @param args Positional value IDs of inputs and outputs for the operator.
   */
  readonly recordIn: (ctx: WgpuExecutionContext, args: TArgs) => void;
};
