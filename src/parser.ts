/**
 * ExecuTorch .pte file format and WebGPU delegate binary parser.
 *
 * Implements parsing of:
 * 1. ExecuTorch container and ExtendedHeader ('eh00', 'ET12')
 * 2. ExecutionPlan and BackendDelegate references
 * 3. WebGPU delegate header ('VH00')
 * 4. VkGraph ('VK00') operator call graph, values, inputs, outputs, and constants
 */

import { ByteBuffer } from 'flatbuffers';
import type { DType } from './tensor';

export const VK_DATA_TYPE_MAP: Record<number, DType> = {
  0: 'bool',
  1: 'uint8',
  2: 'int8',
  3: 'int32',
  4: 'float16',
  5: 'float32',
  6: 'float64',
  7: 'int64',
};

export type ParsedVkTensor = {
  readonly kind: 'tensor';
  readonly dtype: DType;
  readonly dims: readonly number[];
  readonly constantId: number;
  readonly memObjId: number;
};

export type ParsedVkScalar = {
  readonly kind: 'scalar';
  readonly type: 'int' | 'double' | 'bool' | 'symint';
  readonly value: number | boolean;
};

export type ParsedVkString = {
  readonly kind: 'string';
  readonly value: string;
};

export type ParsedVkList = {
  readonly kind: 'list';
  readonly items: readonly (number | boolean)[];
};

export type ParsedVkValue =
  ParsedVkTensor | ParsedVkScalar | ParsedVkString | ParsedVkList | { readonly kind: 'null' };

export type ParsedOperatorCall = {
  readonly name: string;
  readonly args: readonly number[];
};

export type ParsedConstant = {
  readonly offset: number;
  readonly length: number;
  readonly namedKey?: string;
};

export type ParsedDelegateGraph = {
  readonly backendId: string;
  readonly inputIds: readonly number[];
  readonly outputIds: readonly number[];
  readonly values: readonly ParsedVkValue[];
  readonly chain: readonly ParsedOperatorCall[];
  readonly constants: readonly ParsedConstant[];
  readonly constantData: Uint8Array;
};

export type ParsedMethod = {
  readonly name: string;
  readonly delegate: ParsedDelegateGraph;
};

export type ParsedProgram = {
  readonly version: number;
  readonly methods: Map<string, ParsedMethod>;
};

/**
 * ExtendedHeader layout (64 bytes minimum head inspection):
 * 0..4: FlatBuffer root table offset
 * 4..8: File identifier ('ET12' or 'eT01')
 * 8..12: Extended header magic ('eh00')
 * 12..16: header_length (uint32 LE, >= 24)
 * 16..24: program_size (uint64 LE)
 * 24..32: segment_base_offset (uint64 LE)
 * 32..40: segment_data_size (uint64 LE, optional)
 */
export type ExtendedHeader = {
  readonly headerLength: number;
  readonly programSize: number;
  readonly segmentBaseOffset: number;
  readonly segmentDataSize: number;
};

/**
 * WebGPUDelegateHeader layout (minimum 30 bytes):
 * 4..8: Magic ('VH00')
 * 8..10: header_size (uint16 LE, >= 30)
 * 10..14: flatbuffer_offset (uint32 LE)
 * 14..18: flatbuffer_size (uint32 LE)
 * 18..22: bytes_offset (uint32 LE)
 * 22..30: bytes_size (uint64 LE)
 */
export type WebGpuDelegateHeader = {
  readonly headerSize: number;
  readonly flatbufferOffset: number;
  readonly flatbufferSize: number;
  readonly bytesOffset: number;
  readonly bytesSize: number;
};

/**
 * Reads a 64-bit unsigned integer as a JavaScript number.
 *
 * @param bb The flatbuffers ByteBuffer.
 * @param offset The byte offset into the buffer.
 * @returns The uint64 value as a number.
 */
function readUint64Number(bb: ByteBuffer, offset: number): number {
  return Number(bb.readUint64(offset));
}

/**
 * Reads a 64-bit signed integer as a JavaScript number.
 *
 * @param bb The flatbuffers ByteBuffer.
 * @param offset The byte offset into the buffer.
 * @returns The int64 value as a number.
 */
function readInt64Number(bb: ByteBuffer, offset: number): number {
  return Number(bb.readInt64(offset));
}

/**
 * Reads a UTF-8 string from a FlatBuffer table field.
 *
 * @param bb The flatbuffers ByteBuffer.
 * @param offset The byte offset of the string field.
 * @returns The decoded string.
 */
function readString(bb: ByteBuffer, offset: number): string {
  const res = bb.__string(offset);
  return typeof res === 'string' ? res : new TextDecoder('utf-8').decode(res);
}

/**
 * Parses the ExecuTorch ExtendedHeader from the raw binary file.
 *
 * @param bytes The raw file binary bytes.
 * @returns The parsed ExtendedHeader or null if not present.
 */
export function parseExtendedHeader(bytes: Uint8Array): ExtendedHeader | null {
  if (bytes.byteLength < 32) return null;
  const magic = new TextDecoder('ascii').decode(bytes.subarray(8, 12));
  if (magic !== 'eh00') return null;

  const bb = new ByteBuffer(bytes);
  const headerLength = bb.readUint32(12);
  const programSize = readUint64Number(bb, 16);
  const segmentBaseOffset = readUint64Number(bb, 24);

  let segmentDataSize = 0;
  if (headerLength >= 32 && bytes.byteLength >= 40) {
    segmentDataSize = readUint64Number(bb, 32);
  }

  return {
    headerLength,
    programSize,
    segmentBaseOffset,
    segmentDataSize,
  };
}

/**
 * Parses the WebGPUDelegateHeader ('VH00') from raw delegate binary data.
 *
 * @param bytes Raw binary slice containing the WebGPU delegate header.
 * @returns The parsed WebGpuDelegateHeader descriptor.
 */
export function parseWebGpuDelegateHeader(bytes: Uint8Array): WebGpuDelegateHeader {
  if (bytes.byteLength < 30) {
    throw new Error('WebGPU delegate header size too small');
  }
  const magic = new TextDecoder('ascii').decode(bytes.subarray(4, 8));
  if (magic !== 'VH00') {
    throw new Error(`Invalid WebGPU delegate magic (expected 'VH00', got '${magic}')`);
  }
  const bb = new ByteBuffer(bytes);
  const headerSize = bb.readUint16(8);
  const flatbufferOffset = bb.readUint32(10);
  const flatbufferSize = bb.readUint32(14);
  const bytesOffset = bb.readUint32(18);
  const bytesSize = readUint64Number(bb, 22);

  return {
    headerSize,
    flatbufferOffset,
    flatbufferSize,
    bytesOffset,
    bytesSize,
  };
}

/**
 * Decodes a VkGraph ('VK00') FlatBuffer table and its contained execution chain.
 *
 * @param fbBytes FlatBuffer binary slice containing the VkGraph table.
 * @param constantData Contiguous raw weight constant bytes.
 * @param backendId The backend delegate identifier string.
 * @returns Parsed delegate graph execution description.
 */
export function parseVkGraph(
  fbBytes: Uint8Array,
  constantData: Uint8Array,
  backendId: string
): ParsedDelegateGraph {
  const bb = new ByteBuffer(fbBytes);
  if (!bb.__has_identifier('VK00')) {
    throw new Error("Invalid VkGraph magic (expected 'VK00')");
  }

  const root = bb.readUint32(0);

  // Field 1: chain ([OperatorCall]) -> vtable offset 6
  const chainFo = bb.__offset(root, 6);
  const chain: ParsedOperatorCall[] = [];
  if (chainFo !== 0) {
    const chainVec = bb.__vector(root + chainFo);
    const chainLen = bb.__vector_len(root + chainFo);
    for (let i = 0; i < chainLen; i++) {
      const opTable = bb.__indirect(chainVec + i * 4);
      // OperatorCall: name (field 1 -> vo 6), args (field 2 -> vo 8)
      const nameFo = bb.__offset(opTable, 6);
      const opName = nameFo !== 0 ? readString(bb, opTable + nameFo) : '';

      const argsFo = bb.__offset(opTable, 8);
      const args: number[] = [];
      if (argsFo !== 0) {
        const argsVec = bb.__vector(opTable + argsFo);
        const argsLen = bb.__vector_len(opTable + argsFo);
        for (let a = 0; a < argsLen; a++) {
          args.push(bb.readInt32(argsVec + a * 4));
        }
      }
      chain.push({ name: opName, args });
    }
  }

  // Field 2: values ([VkValue]) -> vtable offset 8
  const valuesFo = bb.__offset(root, 8);
  const values: ParsedVkValue[] = [];
  if (valuesFo !== 0) {
    const valuesVec = bb.__vector(root + valuesFo);
    const valuesLen = bb.__vector_len(root + valuesFo);
    for (let i = 0; i < valuesLen; i++) {
      const valTable = bb.__indirect(valuesVec + i * 4);
      // VkValue: type (field 0 -> vo 4), value (field 1 -> vo 6)
      const typeFo = bb.__offset(valTable, 4);
      const valType = typeFo !== 0 ? bb.readUint8(valTable + typeFo) : 0;
      const dataFo = bb.__offset(valTable, 6);

      if (dataFo === 0 || valType === 0) {
        values.push({ kind: 'null' });
        continue;
      }

      const tableOffset = bb.__indirect(valTable + dataFo);
      switch (valType) {
        case 5: {
          // VkTensor: datatype (0 -> vo 4), dims (1 -> vo 6), constant_id (2 -> vo 8), mem_obj_id (3 -> vo 10)
          const dtFo = bb.__offset(tableOffset, 4);
          const dtCode = dtFo !== 0 ? bb.readUint8(tableOffset + dtFo) : 5;
          const dtype = VK_DATA_TYPE_MAP[dtCode] ?? 'float32';

          const dimsFo = bb.__offset(tableOffset, 6);
          const dims: number[] = [];
          if (dimsFo !== 0) {
            const dimsVec = bb.__vector(tableOffset + dimsFo);
            const dimsLen = bb.__vector_len(tableOffset + dimsFo);
            for (let d = 0; d < dimsLen; d++) {
              dims.push(bb.readUint32(dimsVec + d * 4));
            }
          }

          const constFo = bb.__offset(tableOffset, 8);
          const constantId = constFo !== 0 ? bb.readInt32(tableOffset + constFo) : -1;

          const memFo = bb.__offset(tableOffset, 10);
          const memObjId = memFo !== 0 ? bb.readInt32(tableOffset + memFo) : -1;

          values.push({
            kind: 'tensor',
            dtype,
            dims,
            constantId,
            memObjId,
          });
          break;
        }
        case 2: {
          // Int: int_val (0 -> vo 4)
          const valFo = bb.__offset(tableOffset, 4);
          const val = valFo !== 0 ? readInt64Number(bb, tableOffset + valFo) : 0;
          values.push({ kind: 'scalar', type: 'int', value: val });
          break;
        }
        case 3: {
          // Double: double_val (0 -> vo 4)
          const valFo = bb.__offset(tableOffset, 4);
          const val = valFo !== 0 ? bb.readFloat64(tableOffset + valFo) : 0.0;
          values.push({ kind: 'scalar', type: 'double', value: val });
          break;
        }
        case 4: {
          // Bool: bool_val (0 -> vo 4)
          const valFo = bb.__offset(tableOffset, 4);
          const val = valFo !== 0 ? bb.readUint8(tableOffset + valFo) !== 0 : false;
          values.push({ kind: 'scalar', type: 'bool', value: val });
          break;
        }
        case 11: {
          // SymInt: value (0 -> vo 4)
          const valFo = bb.__offset(tableOffset, 4);
          const val = valFo !== 0 ? bb.readInt32(tableOffset + valFo) : 0;
          values.push({ kind: 'scalar', type: 'symint', value: val });
          break;
        }
        case 10: {
          // String: string_val (0 -> vo 4)
          const valFo = bb.__offset(tableOffset, 4);
          const val = valFo !== 0 ? readString(bb, tableOffset + valFo) : '';
          values.push({ kind: 'string', value: val });
          break;
        }
        case 6: {
          // IntList: items (0 -> vo 4)
          const itemsFo = bb.__offset(tableOffset, 4);
          const items: number[] = [];
          if (itemsFo !== 0) {
            const vec = bb.__vector(tableOffset + itemsFo);
            const len = bb.__vector_len(tableOffset + itemsFo);
            for (let k = 0; k < len; k++) {
              items.push(readInt64Number(bb, vec + k * 8));
            }
          }
          values.push({ kind: 'list', items });
          break;
        }
        case 9: {
          // ValueList: items (0 -> vo 4)
          const itemsFo = bb.__offset(tableOffset, 4);
          const items: number[] = [];
          if (itemsFo !== 0) {
            const vec = bb.__vector(tableOffset + itemsFo);
            const len = bb.__vector_len(tableOffset + itemsFo);
            for (let k = 0; k < len; k++) {
              items.push(bb.readInt32(vec + k * 4));
            }
          }
          values.push({ kind: 'list', items });
          break;
        }
        default:
          values.push({ kind: 'null' });
          break;
      }
    }
  }

  // Field 3: input_ids ([uint]) -> vtable offset 10
  const inFo = bb.__offset(root, 10);
  const inputIds: number[] = [];
  if (inFo !== 0) {
    const inVec = bb.__vector(root + inFo);
    const inLen = bb.__vector_len(root + inFo);
    for (let i = 0; i < inLen; i++) {
      inputIds.push(bb.readUint32(inVec + i * 4));
    }
  }

  // Field 4: output_ids ([uint]) -> vtable offset 12
  const outFo = bb.__offset(root, 12);
  const outputIds: number[] = [];
  if (outFo !== 0) {
    const outVec = bb.__vector(root + outFo);
    const outLen = bb.__vector_len(root + outFo);
    for (let i = 0; i < outLen; i++) {
      outputIds.push(bb.readUint32(outVec + i * 4));
    }
  }

  // Field 5: constants ([VkBytes]) -> vtable offset 14
  const constFo = bb.__offset(root, 14);
  const constants: ParsedConstant[] = [];
  if (constFo !== 0) {
    const constVec = bb.__vector(root + constFo);
    const constLen = bb.__vector_len(root + constFo);
    for (let i = 0; i < constLen; i++) {
      const cTable = bb.__indirect(constVec + i * 4);
      // VkBytes: offset (0 -> vo 4), length (1 -> vo 6), named_key (2 -> vo 8)
      const offFo = bb.__offset(cTable, 4);
      const lenFo = bb.__offset(cTable, 6);
      const keyFo = bb.__offset(cTable, 8);

      const offset = offFo !== 0 ? readUint64Number(bb, cTable + offFo) : 0;
      const length = lenFo !== 0 ? readUint64Number(bb, cTable + lenFo) : 0;
      const namedKey = keyFo !== 0 ? readString(bb, cTable + keyFo) : undefined;

      constants.push({ offset, length, namedKey });
    }
  }

  return {
    backendId,
    inputIds,
    outputIds,
    values,
    chain,
    constants,
    constantData,
  };
}

/**
 * Parses an entire ExecuTorch `.pte` model binary into a structured Program.
 *
 * @param pteData Raw Uint8Array of the entire `.pte` file.
 * @returns The parsed Program with all execution plans and WebGPU delegate graphs.
 */
export function parsePte(pteData: Uint8Array): ParsedProgram {
  const bb = new ByteBuffer(pteData);
  const extHeader = parseExtendedHeader(pteData);

  const root = bb.readUint32(0);

  // Program fields in program.fbs:
  // 0: version (uint -> vo 4)
  // 1: execution_plan ([ExecutionPlan] -> vo 6)
  // 2: constant_buffer ([Buffer] -> vo 8)
  // 3: backend_delegate_data ([BackendDelegateInlineData] -> vo 10)
  // 4: segments ([DataSegment] -> vo 12)
  const verFo = bb.__offset(root, 4);
  const version = verFo !== 0 ? bb.readUint32(root + verFo) : 0;

  // Segments table
  const segFo = bb.__offset(root, 12);
  const segments: { readonly offset: number; readonly size: number }[] = [];
  if (segFo !== 0) {
    const segVec = bb.__vector(root + segFo);
    const segLen = bb.__vector_len(root + segFo);
    for (let s = 0; s < segLen; s++) {
      const sTable = bb.__indirect(segVec + s * 4);
      // DataSegment: offset (0 -> vo 4), size (1 -> vo 6)
      const offFo = bb.__offset(sTable, 4);
      const szFo = bb.__offset(sTable, 6);
      const off = offFo !== 0 ? readUint64Number(bb, sTable + offFo) : 0;
      const sz = szFo !== 0 ? readUint64Number(bb, sTable + szFo) : 0;
      segments.push({ offset: off, size: sz });
    }
  }

  // Execution plans
  const epFo = bb.__offset(root, 6);
  if (epFo === 0) {
    throw new Error('PTE program has no execution plans');
  }

  const epVec = bb.__vector(root + epFo);
  const epLen = bb.__vector_len(root + epFo);
  const methods = new Map<string, ParsedMethod>();

  for (let i = 0; i < epLen; i++) {
    const planTable = bb.__indirect(epVec + i * 4);
    // ExecutionPlan fields:
    // 0: name (string -> vo 4)
    // 7: delegates ([BackendDelegate] -> vo 18)
    const nameFo = bb.__offset(planTable, 4);
    const methodName = nameFo !== 0 ? readString(bb, planTable + nameFo) : `method_${i}`;

    const delFo = bb.__offset(planTable, 18);
    if (delFo === 0) continue;

    const delVec = bb.__vector(planTable + delFo);
    const delLen = bb.__vector_len(planTable + delFo);
    if (delLen === 0) continue;

    // We locate the WebGPU or Vulkan delegate
    for (let d = 0; d < delLen; d++) {
      const dTable = bb.__indirect(delVec + d * 4);
      // BackendDelegate fields:
      // 0: id (string -> vo 4)
      // 1: processed (BackendDelegateDataReference -> vo 6)
      const idFo = bb.__offset(dTable, 4);
      const delegateId = idFo !== 0 ? readString(bb, dTable + idFo) : '';

      const procFo = bb.__offset(dTable, 6);
      if (procFo === 0) continue;

      const procTable = bb.__indirect(dTable + procFo);
      // BackendDelegateDataReference: location (0: INLINE=0, SEGMENT=1 -> vo 4), index (1 -> vo 6)
      const locFo = bb.__offset(procTable, 4);
      const idxFo = bb.__offset(procTable, 6);
      const location = locFo !== 0 ? bb.readUint8(procTable + locFo) : 0;
      const segIndex = idxFo !== 0 ? bb.readUint32(procTable + idxFo) : 0;

      let delegateBytes: Uint8Array | null = null;
      if (location === 1) {
        // SEGMENT
        const segBase = extHeader?.segmentBaseOffset ?? 0;
        const targetSeg = segments[segIndex];
        if (targetSeg) {
          const absOffset = segBase + targetSeg.offset;
          delegateBytes = pteData.subarray(absOffset, absOffset + targetSeg.size);
        }
      } else {
        // INLINE: backend_delegate_data field 3 -> vo 10
        const bddFo = bb.__offset(root, 10);
        if (bddFo !== 0) {
          const bddVec = bb.__vector(root + bddFo);
          const bddTable = bb.__indirect(bddVec + segIndex * 4);
          // BackendDelegateInlineData: data (field 0 -> vo 4)
          const dataFo = bb.__offset(bddTable, 4);
          if (dataFo !== 0) {
            const dataVec = bb.__vector(bddTable + dataFo);
            const dataLen = bb.__vector_len(bddTable + dataFo);
            delegateBytes = pteData.subarray(dataVec, dataVec + dataLen);
          }
        }
      }

      if (!delegateBytes || delegateBytes.byteLength < 30) continue;

      // Unpack WebGPUDelegateHeader ('VH00')
      const vh = parseWebGpuDelegateHeader(delegateBytes);
      const vkFbBytes = delegateBytes.subarray(
        vh.flatbufferOffset,
        vh.flatbufferOffset + vh.flatbufferSize
      );
      const constantData = delegateBytes.subarray(
        vh.bytesOffset,
        vh.bytesOffset +
          (vh.bytesSize > 0 ? vh.bytesSize : delegateBytes.byteLength - vh.bytesOffset)
      );

      const parsedDelegate = parseVkGraph(vkFbBytes, constantData, delegateId);
      methods.set(methodName, {
        name: methodName,
        delegate: parsedDelegate,
      });
      break;
    }
  }

  return {
    version,
    methods,
  };
}
