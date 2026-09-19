import type { WgpuExecutionContext } from './context';

/**
 * Represents a GPU compute shader in the ExecuTorch WebGPU execution engine.
 *
 * Each shader encapsulates:
 * 1. Its canonical PyTorch / ExecuTorch operator identifier (`name`, e.g.
 *    `'aten::add.Tensor'`).
 * 2. Its raw WebGPU Shading Language source code (`code`), which can be a single
 *    shader string or a record of named shader variants
 *    (e.g. `{ tiled: string, vec4: string }`).
 * 3. Its record construction logic (`recordIn`), which validates input/output
 *    tensors, allocates shape metadata uniform buffers, compiles or retrieves
 *    cached compute pipelines, and records the resulting compute dispatch into
 *    the given {@link WgpuExecutionContext}.
 * @typeParam TArgs The tuple schema of value IDs accepted by this shader's
 * {@link recordIn} method.
 * @typeParam TName Literal string type of the operator name.
 */
export type Shader<
  TArgs extends readonly PropertyKey[] = readonly PropertyKey[],
  TName extends string = string,
> = {
  readonly name: TName;
  readonly code: string | Record<string, string>;
  readonly recordIn: (ctx: WgpuExecutionContext, args: TArgs) => void;
};
