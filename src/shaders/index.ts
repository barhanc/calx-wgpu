import { add } from './add';
import { mm } from './mm';

/**
 * Namespace containing all registered WebGPU compute shaders.
 */
export const shaders = {
  add,
  mm,
} as const;
