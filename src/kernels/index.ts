import { add } from './add';

export * from './add';

/**
 * Namespace containing all registered WebGPU compute kernels.
 */
export const kernels = {
  add,
} as const;
