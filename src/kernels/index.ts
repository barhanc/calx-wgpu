import { add } from './add';
import { mm } from './mm';

export * from './add';
export * from './mm';

/**
 * Namespace containing all registered WebGPU compute kernels.
 */
export const kernels = {
  add,
  mm,
} as const;
