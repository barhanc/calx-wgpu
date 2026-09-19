import { ByteBuffer } from 'flatbuffers';
import { Program } from './generated/program_generated';
import * as Schema from './generated/schema_generated';
import { DTYPE_BYTESIZE, type DType } from './tensor';

const VK_DTYPE_MAP: Record<number, DType> = {
  [Schema.VkDataType.BOOL]: 'bool',
  [Schema.VkDataType.UINT8]: 'uint8',
  [Schema.VkDataType.INT8]: 'int8',
  [Schema.VkDataType.INT32]: 'int32',
  [Schema.VkDataType.FLOAT16]: 'float16',
  [Schema.VkDataType.FLOAT32]: 'float32',
  [Schema.VkDataType.FLOAT64]: 'float64',
  [Schema.VkDataType.INT64]: 'int64',
};

export type ParsedTensorMeta = {
  kind: 'tensor';
  dtype: DType;
  dims: number[];
  constantId: number;
  memObjId: number;
  byteSize: number;
};

export type ParsedScalarMeta = {
  kind: 'scalar';
  value: number;
};

export type ParsedValue = ParsedTensorMeta | ParsedScalarMeta | { kind: 'other' };

export type ParsedOp = {
  name: string;
  args: number[];
};

/**
 * Clean, non-null wrapper over raw VkGraph FlatBuffer table.
 */
export class DelegateGraph {
  readonly #graph: Schema.VkGraph;
  readonly #constantData: Uint8Array;

  constructor(graph: Schema.VkGraph, constantData: Uint8Array) {
    this.#graph = graph;
    this.#constantData = constantData;
  }

  get valuesCount(): number {
    return this.#graph.valuesLength();
  }

  get opsCount(): number {
    return this.#graph.chainLength();
  }

  get inputIds(): number[] {
    return Array.from(this.#graph.inputIdsArray() ?? []);
  }

  get outputIds(): number[] {
    return Array.from(this.#graph.outputIdsArray() ?? []);
  }

  getValue(index: number): ParsedValue {
    const val = this.#graph.values(index);
    if (!val) return { kind: 'other' };

    const type = val.valueType();
    if (type === Schema.GraphTypes.VkTensor) {
      const t = val.value(new Schema.VkTensor()) as Schema.VkTensor;
      const dims = Array.from(t.dimsArray() ?? []);
      const dtype = VK_DTYPE_MAP[t.datatype()] ?? 'float32';
      const numel = dims.reduce((a, b) => a * b, 1);
      return {
        kind: 'tensor',
        dtype,
        dims,
        constantId: t.constantId(),
        memObjId: t.memObjId(),
        byteSize: Math.max(numel * DTYPE_BYTESIZE[dtype], 4),
      };
    }
    if (type === Schema.GraphTypes.Int) {
      return {
        kind: 'scalar',
        value: Number((val.value(new Schema.Int()) as Schema.Int)?.intVal() ?? 0),
      };
    }
    if (type === Schema.GraphTypes.Double) {
      return {
        kind: 'scalar',
        value: (val.value(new Schema.Double()) as Schema.Double)?.doubleVal() ?? 0,
      };
    }
    if (type === Schema.GraphTypes.Bool) {
      return {
        kind: 'scalar',
        value: (val.value(new Schema.Bool()) as Schema.Bool)?.boolVal() ? 1 : 0,
      };
    }
    if (type === Schema.GraphTypes.SymInt) {
      return {
        kind: 'scalar',
        value: (val.value(new Schema.SymInt()) as Schema.SymInt)?.value() ?? 0,
      };
    }
    return { kind: 'other' };
  }

  getConstant(constantId: number): Uint8Array | null {
    if (constantId < 0 || constantId >= this.#graph.constantsLength()) return null;
    const meta = this.#graph.constants(constantId)!;
    const off = Number(meta.offset());
    const len = Number(meta.length());
    return this.#constantData.subarray(off, off + len);
  }

  getOp(index: number): ParsedOp {
    const op = this.#graph.chain(index)!;
    return {
      name: op.name() ?? '',
      args: Array.from(op.argsArray() ?? []),
    };
  }
}

export function parsePte(bytes: Uint8Array): {
  methodNames: string[];
  getDelegate(planIdx?: number): DelegateGraph;
} {
  const program = Program.getRootAsProgram(new ByteBuffer(bytes));
  const epLen = program.executionPlanLength();
  const methodNames: string[] = [];
  for (let i = 0; i < epLen; i++) {
    methodNames.push(program.executionPlan(i)?.name() ?? `method_${i}`);
  }

  return {
    methodNames,
    getDelegate(planIdx = 0): DelegateGraph {
      const seg = program.segments(
        program.executionPlan(planIdx)!.delegates(0)!.processed()!.index()
      )!;
      const segBase = Number(new ByteBuffer(bytes).readUint64(24));
      const off = segBase + Number(seg.offset());
      const raw = bytes.subarray(off, off + Number(seg.size()));
      const vh = new ByteBuffer(raw);
      const graph = Schema.VkGraph.getRootAsVkGraph(
        new ByteBuffer(raw.subarray(vh.readUint32(10), vh.readUint32(10) + vh.readUint32(14)))
      );
      return new DelegateGraph(graph, raw.subarray(vh.readUint32(18)));
    },
  };
}
