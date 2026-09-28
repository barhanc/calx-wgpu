import { create, globals } from 'webgpu';

// Set up Dawn WebGPU globals before any source file is loaded.
// This must run before src/tensor.ts which uses GPUBufferUsage at module level.
Object.assign(globalThis, globals);

// Create a global navigator with WebGPU for Node.js.
// navigator is read-only on globalThis in Node.js, so use defineProperty.
Object.defineProperty(globalThis, 'navigator', {
  value: { gpu: create([]) },
  writable: true,
  configurable: true,
});
