import { add } from './add';
import { constantPadNd } from './constantPadNd';
import { convolution } from './convolution';
import { mm } from './mm';

/**
 * Namespace containing all registered WebGPU compute shaders.
 */
// prettier-ignore
export const shaders = {
  add, constantPadNd, convolution, mm,
} as const;

/**
 * Maps ExecuTorch operator names to their shader implementations.
 *
 * @internal
 */
export const shaderRegistry = Object.fromEntries(
  Object.values(shaders).map((shader) => [shader.name, shader])
);
