---
name: executorch-webgpu
description: >-
  Reference and development guide for the ExecuTorch WebGPU backend in third-party/executorch/backends/webgpu.
  Use when analyzing ExecuTorch delegate formats, WebGPU ops, WGSL kernels, TensorMeta layout, or porting operators to phlox-wgpu.
---

# ExecuTorch WebGPU Backend Skill

This skill provides reference documentation and guidelines for understanding and porting operators from the ExecuTorch WebGPU backend (`third-party/executorch/backends/webgpu`) into `phlox-wgpu`.

## Key Directory Structure

Path: `third-party/executorch/backends/webgpu/`

- `runtime/WebGPUGraph.h`: Graph container, `WebGPUTensor` definition, and delegate execution model.
- `runtime/ops/TensorMeta.h`: std140 uniform buffer layout (`sizes`, `strides`, `ndim`, `numel`).
- `runtime/ops/`: C++ operator wrappers and WGSL compute shaders (e.g. `add/`, `mm/`, `bmm/`, `conv/`, etc.).
- `runtime/ops/OperatorRegistry.h`: Mapping of ExecuTorch operator schemas to their WebGPU kernels.
- `docs/source/backends/webgpu/webgpu-op-support.md`: Supported operator table and backend status.

## Data Representation: `WebGPUTensor`

In ExecuTorch WebGPU:

- Tensors are contiguous memory blocks backed by a `WGPUBuffer`.
- Only `dims` (shape) and `nbytes` (total byte length) are tracked.
- Strides are **not** stored as properties on the tensor struct.

## Uniform Metadata: `TensorMeta`

All WGSL elementwise and broadcast operations rely on an 80-byte std140 uniform buffer defined in `runtime/ops/TensorMeta.h`:

```wgsl
struct TensorMeta {
  ndim: u32,
  numel: u32,
  sizes: array<vec4<u32>, 2>,
  strides: array<vec4<u32>, 2>,
}
```

- Max rank is 8 (`kTensorMetaMaxNdim = 8`).
- Layout offsets: `ndim` at byte 0, `numel` at byte 4, `sizes` at byte 16, `strides` at byte 48.
- Stride values are computed on-the-fly dynamically when recording dispatches (right-aligned for broadcasting).

## Delegate Serialization

1. ExecuTorch container file (`.pte`) packs execution plans and backend delegates.
2. Delegate header is marked by magic `VH00`.
3. Operator call graph is serialized as a FlatBuffer marked by magic `VK00` (`VkGraph`).
4. Operators are listed in `chain: [OperatorCall]`, mapping operator names (e.g. `aten.add.Tensor`, `aten.mm.default`) to context value indices in `values: [VkValue]`.
