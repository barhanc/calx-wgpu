# WebGPU for ML Inference — Fundamentals

## What is WebGPU?

WebGPU is a low-level GPU API for the web — the successor to WebGL. It gives direct access to GPU compute through a C-like API modeled after Vulkan, Metal, and D3D12. Shaders are written in **WGSL** (WebGPU Shading Language), a Rust-like language compiled by the browser to each platform's native shader language.

The design philosophy: **explicit over implicit**. You manage buffers, pipelines, and command encoding yourself. Everything is validated at creation time — no hidden state machines, no silent corruption.

For ML inference, the relevant subset is small: **storage buffers** hold tensor data, **compute shaders** run the math, and **command encoding** sequences the operations.

---

## Computation Model

WebGPU executes work through **deferred command encoding**. Nothing runs on the GPU until you explicitly submit. This lets the driver optimize an entire batch of operations at once.

```
JavaScript timeline:
  1. Create buffers, pipelines, bind groups    (setup)
  2. Encode commands into a command buffer      (recording)
  3. Submit to the GPU queue                    (execution)
  4. Read results back                          (synchronization)
```

### Three levels of operations

Commands live at different levels and **can't be freely mixed**:

```
QUEUE-LEVEL    — immediate, ordered by the queue
  queue.writeBuffer(buffer, offset, data)     CPU → GPU upload
  queue.submit([commandBuffer])               execute encoded work
  queue.onSubmittedWorkDone()                 Promise fence

ENCODER-LEVEL  — recorded into a command buffer
  encoder.copyBufferToBuffer(src, dst, size)  GPU → GPU copy
  encoder.clearBuffer(buffer)                 zero-fill
  encoder.beginComputePass() → pass           start a compute pass

PASS-LEVEL     — only inside a compute pass
  pass.setPipeline(pipeline)                  bind shader
  pass.setBindGroup(0, bindGroup)             bind resources
  pass.dispatchWorkgroups(x, y, z)            launch compute work
  pass.end()                                  close the pass
```

**The critical rule:** `copyBufferToBuffer` must be **outside** a pass. `dispatchWorkgroups` must be **inside** a pass. Interleaving copies and dispatches requires ending/starting passes:

```typescript
encoder.copyBufferToBuffer(input, internal);    // outside pass
const pass = encoder.beginComputePass();
  pass.dispatchWorkgroups(...);                  // inside pass
  pass.dispatchWorkgroups(...);                  // inside pass
pass.end();
encoder.copyBufferToBuffer(internal, output);   // outside pass
```

All commands in a single `queue.submit()` execute **in order**. The GPU may pipeline internally, but memory effects are ordered — a copy at position 5 sees the results of dispatches at positions 2–3.

---

## Buffers

A `GPUBuffer` is a raw VRAM allocation — the fundamental data structure for ML inference. Tensor data lives in buffers.

### Usage flags

A buffer's `usage` bitfield restricts what operations can target it (validated at creation):

| Flag       | Purpose                                          |
| ---------- | ------------------------------------------------ |
| `STORAGE`  | Read/write from compute shaders (`var<storage>`) |
| `UNIFORM`  | Read-only, small, cached params (`var<uniform>`) |
| `COPY_SRC` | Source of `copyBufferToBuffer`                   |
| `COPY_DST` | Destination of copy / `writeBuffer`              |
| `MAP_READ` | CPU readback via `mapAsync`                      |

Common patterns:

```typescript
// Tensor data (compute input/output)
GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

// Per-dispatch params (shapes, scalars)
GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

// Staging buffer for GPU → CPU readback
GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
```

### Uniform vs Storage

- **Uniform buffers** — max 64KB, read-only in shaders, cached by the GPU. For per-dispatch parameters. Stricter layout (16-byte array stride).
- **Storage buffers** — up to 128MB per binding, read or read-write. For tensor data. Natural layout.

### Data transfer

```typescript
// CPU → GPU (immediate, ordered by queue)
device.queue.writeBuffer(buffer, 0, typedArray);

// GPU → GPU (deferred, encoded into command buffer)
encoder.copyBufferToBuffer(src, 0, dst, 0, size);

// GPU → CPU (requires staging buffer with MAP_READ)
const staging = device.createBuffer({ size, usage: MAP_READ | COPY_DST });
encoder.copyBufferToBuffer(gpuBuf, 0, staging, 0, size);
device.queue.submit([encoder.finish()]);
await staging.mapAsync(GPUMapMode.READ);
const data = staging.getMappedRange(0, size).slice(0); // copy out
staging.destroy();
```

`writeBuffer` is ordered with respect to `submit` — no deferred encoding needed. `copyBufferToBuffer` is encoder-level and participates in the command buffer's ordering.

---

## Compute Shaders (WGSL)

### The mental model

A compute shader is a function that runs on **thousands of threads** in parallel. Each thread executes the same code but with a different **thread ID**. The GPU organizes threads into **workgroups** — groups of threads that can share memory and synchronize.

```
dispatchWorkgroups(64, 1, 1)   →  64 workgroups
  workgroup_size(256, 1, 1)    →  256 threads per workgroup
  total                        →  16,384 threads
```

This is the same model as CUDA (grid/block) or Vulkan (dispatch/local size).

### Thread identity (builtins)

Each thread knows its position via builtins:

```wgsl
@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(global_invocation_id)    gid:  vec3<u32>,  // global thread ID
  @builtin(local_invocation_id)     lid:  vec3<u32>,  // thread ID in workgroup
  @builtin(workgroup_id)            wid:  vec3<u32>,  // workgroup ID
  @builtin(num_workgroups)          nwg:  vec3<u32>,  // dispatch dimensions
) {
  let idx = gid.x;  // linear index for 1D dispatch
  // ... process data[idx] ...
}
```

The key relationship:

```
global_invocation_id = workgroup_id * workgroup_size + local_invocation_id
```

### Address spaces (where data lives)

```wgsl
@group(0) @binding(0) var<storage, read>       input:  array<f32>;  // GPU buffer
@group(0) @binding(1) var<storage, read_write> output: array<f32>;  // GPU buffer
@group(0) @binding(2) var<uniform>             params: Params;       // GPU buffer (small)
var<workgroup> tile: array<array<f32, 32>, 32>;                      // shared memory
```

- **`storage`** — large GPU buffers (tensor data). `read` or `read_write`.
- **`uniform`** — small, read-only, cached (parameters). Layout has extra padding rules.
- **`workgroup`** — shared across threads in one workgroup (like CUDA `__shared__`). Limited to 16KB.

### Shared memory and barriers

Shared memory lets threads in a workgroup cooperate on data (e.g., matrix multiplication tiles). But it requires synchronization — threads must wait for each other before reading shared data:

```wgsl
var<workgroup> tile: array<array<f32, 32>, 32>;

// Load phase: each thread loads a piece
tile[lid.x][lid.y] = input[...];
workgroupBarrier();  // wait for ALL threads to finish loading

// Compute phase: safe to read the full tile
for (var k = 0u; k < 32u; k++) {
  acc += tile[lid.x][k] * tile[k][lid.y];
}
workgroupBarrier();  // wait for ALL threads to finish reading before overwriting
```

`workgroupBarrier()` does three things:

1. All writes before the barrier complete and become visible to all threads in the workgroup
2. All threads wait for each other to arrive
3. All threads resume past the barrier together

**Barriers only synchronize within a single workgroup.** Workgroups execute independently. For cross-workgroup communication, use atomics.

### Override constants (pipeline specialization)

Values set at pipeline creation, not per-dispatch. The compiler specializes the shader (unrolls loops, folds conditionals):

```wgsl
override wg_size: u32 = 256u;
override alpha: f32 = 1.0;
```

```typescript
const pipeline = device.createComputePipeline({
  layout: 'auto',
  compute: {
    module,
    entryPoint: 'main',
    constants: { wg_size: 128, alpha: 2.0 },
  },
});
```

---

## Memory Layout

WGSL has strict layout rules. Getting alignment wrong causes **silent data corruption** — no errors, just garbage.

### The `vec3` trap

`vec3<f32>` has **size 12** but **alignment 16**. In arrays, each element takes 16 bytes (12 data + 4 padding):

```wgsl
struct Tight {
  a: vec3<f32>,  // offset 0,  size 12
  b: f32,        // offset 12, size 4   ← packs into the padding
};

struct Padded {
  a: vec3<f32>,  // offset 0,  size 12
  b: vec3<f32>,  // offset 16, size 12  ← 4-byte gap at offset 12–15
};

array<vec3<f32>, N>  // stride = 16, not 12!
```

**Fix:** use `vec4<f32>` and put useful data in the 4th component.

### Size and alignment table

| Type                    |                Align |                                    Size |
| ----------------------- | -------------------: | --------------------------------------: |
| `f32`, `i32`, `u32`     |                    4 |                                       4 |
| `f16`                   |                    2 |                                       2 |
| `vec2<f32>`             |                    8 |                                       8 |
| `vec3<f32>`             |               **16** |                                  **12** |
| `vec4<f32>`             |                   16 |                                      16 |
| `mat3x3<f32>`           |                   16 |            **48** (3 × 16-byte columns) |
| `mat4x4<f32>`           |                   16 |                                      64 |
| `array<E, N>` (storage) |         `alignOf(E)` |    `N × roundUp(alignOf(E), sizeOf(E))` |
| `array<E, N>` (uniform) |               **16** |            `N × roundUp(16, sizeOf(E))` |
| `struct`                | `max(member aligns)` | `roundUp(structAlign, endOfLastMember)` |

**Uniform buffers** have stricter rules: array element stride must be a multiple of 16 bytes. Use `vec4` slots to avoid padding waste.

---

## Dispatch Model

### Sizing a dispatch

```
total_threads = dispatchX × dispatchY × dispatchZ × wgSizeX × wgSizeY × wgSizeZ
```

For N elements with 1D workgroups of size 256:

```typescript
const wgSize = 256;
const totalWGs = Math.ceil(N / wgSize);
const dispatchX = Math.min(totalWGs, 65535); // max per dimension
const dispatchY = Math.ceil(totalWGs / 65535); // fold overflow into Y
```

### Bounds checking

Always guard against out-of-bounds threads:

```wgsl
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= arrayLength(&data)) { return; }
  data[idx] = compute(data[idx]);
}
```

---

## Synchronization

### The two levels

| Level           | Mechanism                                          | Purpose                         |
| --------------- | -------------------------------------------------- | ------------------------------- |
| **Shader-side** | `workgroupBarrier()`, `storageBarrier()`           | Sync threads within a workgroup |
| **Host-side**   | `queue.onSubmittedWorkDone()`, `buffer.mapAsync()` | Sync CPU with GPU               |

These operate at completely different levels. Shader barriers never affect the CPU; host fences never affect shader threads.

### The ML inference pipeline

```
1. Upload inputs          queue.writeBuffer()          (CPU → GPU)
2. Copy weights           encoder.copyBufferToBuffer() (if needed)
3. Dispatch ops           pass.dispatchWorkgroups()    (GPU compute)
4. Copy outputs           encoder.copyBufferToBuffer() (GPU → staging)
5. Read results           buffer.mapAsync()            (GPU → CPU)
```

All of 2–4 happen in a single `queue.submit()`. Step 5 blocks until the GPU finishes.

### Reading GPU data back

The GPU can't write directly to CPU memory. Use a staging buffer:

```typescript
const staging = device.createBuffer({ size, usage: MAP_READ | COPY_DST });
const encoder = device.createCommandEncoder();
encoder.copyBufferToBuffer(gpuBuf, 0, staging, 0, size);
device.queue.submit([encoder.finish()]);
await staging.mapAsync(GPUMapMode.READ);
const data = staging.getMappedRange(0, size).slice(0);
staging.destroy();
```

---

## Limits That Matter for ML

| Limit                               | Default | Why it matters                              |
| ----------------------------------- | ------: | ------------------------------------------- |
| `maxStorageBufferBindingSize`       |  128 MB | Max tensor size per binding                 |
| `maxBufferSize`                     |  256 MB | Max single buffer allocation                |
| `maxComputeInvocationsPerWorkgroup` |     256 | Max threads per workgroup (CUDA block size) |
| `maxComputeWorkgroupStorageSize`    |   16 KB | Shared memory per workgroup                 |
| `maxComputeWorkgroupsPerDimension`  |  65,535 | Max dispatch dimension                      |
| `maxStorageBuffersPerShaderStage`   |       8 | Max storage buffers per shader              |
| `maxUniformBufferBindingSize`       |   64 KB | Max params buffer size                      |

These are **defaults** — query `device.limits` and request higher via `requiredLimits`. For large models, `maxStorageBufferBindingSize` and `maxBufferSize` are the usual bottlenecks.

---

## Features for ML

| Feature           | Why it matters                                   |
| ----------------- | ------------------------------------------------ |
| `shader-f16`      | Halve memory bandwidth with `f16` tensors        |
| `subgroups`       | Warp-level parallelism (like CUDA warp shuffles) |
| `timestamp-query` | Profile individual kernel execution times        |

---

## Pipeline Caching

Pipeline creation is expensive (shader compilation). Cache by shader source + constants:

```typescript
const cache = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();

function getPipeline(device: GPUDevice, code: string, key: string) {
  let m = cache.get(device) ?? new Map();
  cache.set(device, m);
  if (!m.has(key)) {
    const module = device.createShaderModule({ code });
    m.set(
      key,
      device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      })
    );
  }
  return m.get(key)!;
}
```

---

## Device Lifecycle

```typescript
// Create
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
const device = await adapter.requestDevice({
  requiredFeatures: ['shader-f16', 'subgroups'],
  requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize },
});

// Handle GPU resets
device.lost.then((info) => {
  console.error(`Device lost: ${info.reason} — ${info.message}`);
});

// Validate resource creation
device.pushErrorScope('validation');
const err = await device.popErrorScope();
if (err) console.error(err.message);
```
