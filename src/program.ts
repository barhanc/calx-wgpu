import type { Tensor, DType } from './tensor';

/**
 * A tensor value in a serialized program — references a GPU buffer
 * (constant, shared pool, or dedicated) with shape and dtype metadata.
 */
export type ProgramTensor = {
  readonly type: 'tensor';
  readonly shape: readonly number[];
  readonly dtype: DType;
  /** Shared memory pool index (absent = dedicated buffer). */
  readonly memObjId?: number;
  /** Byte offset into weights.bin (absent = not a constant). */
  readonly weightsOffset?: number;
  /** Byte length in weights.bin (absent = not a constant). */
  readonly weightsLength?: number;
};

/** A scalar value (number, boolean, or string) passed to operators. */
export type ProgramScalar = {
  readonly type: 'scalar';
  readonly value: number | boolean | string;
};

/** An integer list (e.g., shape hints, padding, stride). */
export type ProgramList = {
  readonly type: 'list';
  readonly items: readonly number[];
};

/** An empty/unused value slot. */
export type ProgramNull = {
  readonly type: 'null';
};

/** Union of all value types in a serialized program. */
export type ProgramValue = ProgramTensor | ProgramScalar | ProgramList | ProgramNull;

/** A single operator call — `args` are value indices into `values[]`. */
export type ProgramOp = {
  readonly name: string;
  readonly args: readonly number[];
};

/** A memory pool shared by tensors with non-overlapping lifetimes. */
export type ProgramPool = {
  readonly id: number;
  readonly size: number;
};

/** The serialized program format (program.json). */
export type Program = {
  readonly version: string;
  readonly chain: readonly ProgramOp[];
  readonly values: readonly ProgramValue[];
  readonly inputIds: readonly number[];
  readonly outputIds: readonly number[];
  readonly memoryPlan: {
    readonly pools: readonly ProgramPool[];
  };
};

/**
 * Resolves an operator's value-index args into concrete values.
 *
 * @param op The operator whose args to resolve.
 * @param values The full value table.
 * @param tensors Map from value index to its GPU Tensor view.
 * @returns Resolved args (Tensor, scalar, number[], or undefined).
 */
export function resolveArgs(
  op: ProgramOp,
  values: readonly ProgramValue[],
  tensors: Map<number, Tensor>
): unknown[] {
  return op.args.map((idx) => {
    const v = values[idx];
    switch (v.type) {
      case 'tensor': {
        const t = tensors.get(idx);
        if (t === undefined) {
          throw new Error(`program: no tensor for value ${idx} referenced by '${op.name}'`);
        }
        return t;
      }
      case 'scalar':
        return v.value;
      case 'list':
        return [...v.items];
      case 'null':
        return undefined;
    }
  });
}
