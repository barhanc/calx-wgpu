import { add } from './add';
import { mm } from './mm';

/**
 * Namespace containing all registered WebGPU compute shaders.
 */
export const shaders = {
  add,
  mm,
} as const;

/**
 * Maps ExecuTorch operator names to their shader implementations.
 *
 * @internal
 */
export const shaderRegistry = Object.fromEntries(
  Object.values(shaders).map((shader) => [shader.name, shader])
);
