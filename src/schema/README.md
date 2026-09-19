# Generated FlatBuffers TypeScript Schemas

This directory contains auto-generated TypeScript accessor classes compiled from ExecuTorch FlatBuffer schemas using the `flatc` compiler.

## Source Schemas

The schemas are sourced from `third-party/executorch`:
1. `third-party/executorch/schema/scalar_type.fbs`
2. `third-party/executorch/schema/program.fbs`
3. `third-party/executorch/backends/vulkan/serialization/schema.fbs`

## How to Regenerate

Prerequisites:
- `flatc` (version >= 2.0.8, available via `sudo apt install flatbuffers-compiler` or downloaded from FlatBuffers GitHub releases).

Run the following command from the repository root:

```bash
npm run codegen:schema
```

or directly:

```bash
flatc --ts -o src/schema \
  -I third-party/executorch/schema \
  third-party/executorch/schema/scalar_type.fbs \
  third-party/executorch/schema/program.fbs \
  third-party/executorch/backends/vulkan/serialization/schema.fbs
```

## Structure

- `program_generated.ts` / `executorch-flatbuffer/`: ExecuTorch container schemas (`Program`, `ExecutionPlan`, `BackendDelegate`, `DataSegment`, etc.).
- `schema_generated.ts` / `vkgraph/`: WebGPU/Vulkan delegate operator graph schemas (`VkGraph`, `OperatorCall`, `VkTensor`, `VkValue`, etc.).
- `scalar_type_generated.ts`: Standard PyTorch / ExecuTorch `ScalarType` enum mappings.
