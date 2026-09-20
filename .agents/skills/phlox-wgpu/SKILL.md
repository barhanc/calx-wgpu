---
name: calx-wgpu
description: >-
  Architecture, development guidelines, project scope, and next steps for the calx-wgpu project.
  Use when analyzing the runtime design, Tensor and context APIs, ExecuTorch delegate handling, or continuing development.
---

# Calx WebGPU (`calx-wgpu`) Skill

## Project Mission & Scope

`calx-wgpu` is a pure TypeScript / native WebGPU inference runtime for PyTorch ExecuTorch models (`.pte`).
It executes PyTorch computational graphs and delegates directly in web browsers and WebGPU-enabled JS runtimes without C++ compilation, Emscripten, or WebAssembly overhead.

### Key Tenets

1. **Pure WebGPU**: Written strictly with TypeScript and native WGSL compute shaders.
2. **ExecuTorch Binary Compatibility**: Reads standard `.pte` files (`ET12`, `eh00`, `VH00`, `VK00`) produced by PyTorch ExecuTorch.
3. **Minimal Dependencies**: Only depends on `flatbuffers` for binary parsing.
4. **Zero-Allocation Execution**: Dispatches are recorded once into a `WgpuExecutionContext` and re-submitted repeatedly without recreating buffers or pipelines.

---

## Directory & Architecture Map

- `src/tensor.ts`: `Tensor` class managing VRAM `GPUBuffer`, `dtype`, `shape`, `nbytes`, `numel`, and `#destroyed` guards.
- `src/context.ts`: `WgpuExecutionContext` managing dispatches, uniform/storage buffers, and submitted GPU work.
- `src/dispatch.ts`: `createComputeBundle` and per-device compute pipeline caching.
- `src/kernel.ts`: Kernel registration contracts (`Kernel<TArgs>`).
- `src/kernels/`:
  - `add.ts`: `aten.add.Tensor` with right-aligned rank-8 broadcasting.
  - `mm.ts`: `aten.mm.default` with 32x32 tiled 128-bit `vec4` GEMM + scalar fallback.
  - `utils/broadcast.ts`: Multidimensional right-aligned broadcasting math.
  - `utils/meta.ts`: ExecuTorch std140 80-byte `TensorMeta` serialization.
- `src/parser.ts`: FlatBuffer container parser for `.pte` files and `VK00` delegate operator graphs.
- `src/model.ts`: High-level `Model` class for loading models and executing methods.
- `example/main.ts`: Interactive test harness running full GPU verification suites in browser.

---

## Current Status & Next Steps

### Completed & Ship-shape

- Device acquisition & feature detection (`src/device.ts`).
- VRAM Tensor management (`src/tensor.ts`).
- Context allocation, recording, and multi-submit execution (`src/context.ts`).
- Pipeline caching & bundle management (`src/dispatch.ts`).
- WGSL Add & Tiled Matmul kernels with std140 metadata encoding (`src/kernels/`).
- ESLint JSDoc configured to match `third-party/react-native-executorch` with 0 warnings.

### Next Steps (Immediate Priorities)

1. **Refactor `src/parser.ts`**:
   - Improve type safety and error reporting for invalid/corrupt `.pte` containers.
   - Clean up delegate identification and multiple method plan support.
2. **Refactor `src/model.ts`**:
   - Align execution API with `react-native-executorch`:
     - Explicit input/output shape & dtype validation against the delegate graph before GPU dispatch.
     - Support caller-provided output `Tensor`s (zero allocation inference).
     - Clean lifecycle handling (`model.dispose()`, `#disposed` checks).
3. **Operator Library Expansion**:
   - Port operators from `third-party/executorch/backends/webgpu/runtime/ops/`:
     - Activations: `relu`, `gelu`, `silu`, `sigmoid`, `tanh`.
     - Math: `sub`, `mul`, `div`.
     - Matmul & Attention: `bmm`, `softmax`, `layer_norm`.
     - Structural: `cat`, `view`, `slice`.
