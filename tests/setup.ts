import { create, globals } from 'webgpu';

// Set up Dawn WebGPU globals before any source file is loaded.
// This must run before src/tensor.ts which uses GPUBufferUsage at module level.
Object.assign(globalThis, globals);

// Create a global navigator with WebGPU for Node.js.
// Use SwiftShader for headless environments (CI) where no GPU is available.
// navigator is read-only on globalThis in Node.js, so use defineProperty.
Object.defineProperty(globalThis, 'navigator', {
  value: { gpu: create(['--enable-unsafe-swiftshader=true']) },
  writable: true,
  configurable: true,
});
