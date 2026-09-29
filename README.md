<div align="center">
  <img src="assets/logo.png" alt="Calx logo" width="420">
  <br>
  <img src="assets/title.svg" alt="CALX" width="320">
</div>

---

Calx is a pure WebGPU inference runtime for PyTorch ExecuTorch `.pte` models,
written in TypeScript. Kernels are hand-written WGSL; tensors live in GPU
buffers and are executed as recorded command sequences.

## Status

> I got bored — this was an excursion into WebGPU-based ML inference engines,
> made for learning purposes. I no longer work on this (maybe I will come back)
> because I got bored and pursue other interests.

## Quick start

```bash
npm install
npm run build   # bundle the library
npm test        # vitest + headless WebGPU (SwiftShader)
npm run example # browser demo at localhost
```

```ts
import { WgpuExecutionContext, shaders } from 'calx-wgpu';

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

const ctx = new WgpuExecutionContext(device);

const xData = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
const yData = new Float32Array([10.0, 20.0, 30.0]);

// Create tensors via context factory and upload initial data
const in1 = ctx.tensor('float32', [2, 3]).setData(xData);
const in2 = ctx.tensor('float32', [1, 3]).setData(yData);
const out = ctx.tensor('float32', [2, 3]);

// Compute out = in1 + 2.0 * in2
ctx.recordShader(shaders.add, [in1, in2, 2.0, out]);

const t0 = performance.now();
ctx.submit();
ctx.sync();
const durMs = performance.now() - t0;

const outBytes = await out.getData();
const outArray = new Float32Array(outBytes);
```
