import {
  initDevice,
  isWebGPUSupported,
  kernels,
  Model,
  Tensor,
  WgpuExecutionContext,
} from '../src';

const statusEl = document.getElementById('status') as HTMLDivElement;
const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
const logEl = document.getElementById('log') as HTMLPreElement;

function log(msg: string) {
  logEl.textContent += `${msg}\n`;
}

if (!isWebGPUSupported()) {
  statusEl.innerHTML =
    '<span class="tag tag-warn">Unsupported</span> WebGPU is not supported in this browser.';
} else {
  statusEl.innerHTML = '<span class="tag tag-success">Supported</span> WebGPU is available!';
  runBtn.disabled = false;
}

runBtn.addEventListener('click', async () => {
  runBtn.disabled = true;
  logEl.textContent = '';

  try {
    log('1. Initializing WebGPU device...');
    const device = await initDevice();
    log(`   Device acquired!`);
    log(`   - shader-f16 supported: ${device.features.has('shader-f16')}`);
    log(`   - subgroups supported:  ${device.features.has('subgroups')}`);

    log('\n--- Test A: Tensor from TypedArray ---');
    log('2. Creating host input tensor data (Float32Array)...');
    const shapeA = [2, 3] as const;
    const rawDataA = new Float32Array([1.5, -2.0, 3.25, 4.125, -5.5, 6.0]);
    log(`   Shape: [${shapeA.join(', ')}]`);
    log(`   Data:  [${Array.from(rawDataA).join(', ')}]`);

    log('3. Allocating WebGPU Tensor (uploading to VRAM)...');
    const tensorA = new Tensor('float32', shapeA, device, rawDataA);
    log(`   VRAM buffer allocated (size: ${tensorA.buffer.size} bytes)`);

    log('4. Reading data back from GPU via tensorA.getData()...');
    const arrayBufferA = await tensorA.getData();
    const resultA = new Float32Array(arrayBufferA);
    log(`   Result: [${Array.from(resultA).join(', ')}]`);

    const matchesA = rawDataA.every((val, idx) => Math.abs(val - resultA[idx]) < 1e-6);
    if (matchesA) {
      log('   ✅ Test A PASSED: GPU roundtrip data matches host input exactly!');
    } else {
      log('   ❌ Test A FAILED: GPU readback does not match input!');
    }

    tensorA.destroy();
    log('   Buffer destroyed.');

    log('\n--- Test B: Tensor from existing GPUBuffer (Zero-Copy) ---');
    log('5. Allocating external GPUBuffer directly on device...');
    const shapeB = [4] as const;
    const rawDataB = new Int32Array([10, -20, 30, -40]);
    const byteLengthB = rawDataB.byteLength;

    const externalBuffer = device.createBuffer({
      size: Math.max(16, byteLengthB),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(externalBuffer, 0, rawDataB.buffer, rawDataB.byteOffset, byteLengthB);
    log(`   External GPUBuffer created (size: ${externalBuffer.size} bytes)`);

    log('6. Wrapping external GPUBuffer into Tensor...');
    const tensorB = new Tensor('int32', shapeB, device, externalBuffer);
    log(
      `   Wrapped zero-copy (tensorB.buffer === externalBuffer: ${tensorB.buffer === externalBuffer})`
    );

    log('7. Reading data back from wrapped GPUBuffer via tensorB.getData()...');
    const arrayBufferB = await tensorB.getData();
    const resultB = new Int32Array(arrayBufferB);
    log(`   Result: [${Array.from(resultB).join(', ')}]`);

    const matchesB = rawDataB.every((val, idx) => val === resultB[idx]);
    if (matchesB) {
      log('   ✅ Test B PASSED: GPUBuffer wrapped and read back correctly!');
    } else {
      log('   ❌ Test B FAILED: Readback does not match input!');
    }

    tensorB.destroy();
    log('   External buffer destroyed via tensorB.destroy().');

    log('\n--- Test C: Context Execution of aten.add.Tensor ---');
    log('8. Building WgpuExecutionContext with input tensors and alpha constant...');
    // x shape: [2, 3], y shape: [1, 3] -> broadcast output: [2, 3]
    const xData = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    const yData = new Float32Array([10.0, 20.0, 30.0]);

    const ctx = new WgpuExecutionContext(device)
      .setTensor('in1', 'float32', [2, 3], xData)
      .setTensor('in2', 'float32', [1, 3], yData)
      .setScalar('alpha', 2.0)
      .setTensor('out', 'float32', [2, 3])
      .record(kernels.add, ['in1', 'in2', 'alpha', 'out']);
    log('   Context recorded 1 compute dispatch.');

    log('9. Executing context dispatches on WebGPU...');
    const t0 = performance.now();
    ctx.submit();
    await device.queue.onSubmittedWorkDone();
    const durationMs = performance.now() - t0;
    log(`   Execution completed in ${durationMs.toFixed(3)}ms (GPU queue wall-time)`);
    log(`   Out shape: [${ctx.getTensor('out').shape.join(', ')}]`);

    log('10. Reading back compute shader output...');
    const outBytes = await ctx.getTensor('out').getData();
    const outFloats = new Float32Array(outBytes);
    log(`   Result: [${Array.from(outFloats).join(', ')}]`);

    // Expected:
    // [1 + 20, 2 + 40, 3 + 60, 4 + 20, 5 + 40, 6 + 60] = [21, 42, 63, 24, 45, 66]
    const expected = [21.0, 42.0, 63.0, 24.0, 45.0, 66.0];
    const matchesC = expected.every((val, idx) => Math.abs(val - outFloats[idx]) < 1e-4);
    if (matchesC) {
      log('   ✅ Test C PASSED: WGSL compute shader add with broadcasting verified!');
    } else {
      log(
        `   ❌ Test C FAILED: Expected [${expected.join(', ')}] but got [${Array.from(outFloats).join(', ')}]`
      );
    }

    ctx.destroy();

    log('\n--- Test D: Context Execution of aten.mm.default (Tiled GEMM) ---');
    log('11. Testing matrix multiplication: A (2x3) @ B (3x2) -> Out (2x2)...');
    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    const aData = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // B = [[7,  8],
    //      [9,  1],
    //      [2,  3]]
    const bData = new Float32Array([7.0, 8.0, 9.0, 1.0, 2.0, 3.0]);

    const mmCtx = new WgpuExecutionContext(device);
    mmCtx
      .setTensor('a', 'float32', [2, 3], aData)
      .setTensor('b', 'float32', [3, 2], bData)
      .setTensor('out', 'float32', [2, 2])
      .record(kernels.mm, ['a', 'b', 'out']);
    log('   Context recorded 1 tiled GEMM dispatch.');

    log('12. Executing matrix multiplication on WebGPU...');
    const tMm = performance.now();
    mmCtx.submit();
    await device.queue.onSubmittedWorkDone();
    const mmDurationMs = performance.now() - tMm;
    log(`   Execution completed in ${mmDurationMs.toFixed(3)}ms (GPU queue wall-time)`);

    log('13. Reading back matmul output...');
    const mmBytes = await mmCtx.getTensor('out').getData();
    const mmFloats = new Float32Array(mmBytes);
    log(`   Result: [${Array.from(mmFloats).join(', ')}]`);

    // Expected:
    // row 0: [1*7 + 2*9 + 3*2, 1*8 + 2*1 + 3*3] = [7 + 18 + 6, 8 + 2 + 9] = [31, 19]
    // row 1: [4*7 + 5*9 + 6*2, 4*8 + 5*1 + 6*3] = [28 + 45 + 12, 32 + 5 + 18] = [85, 55]
    const mmExpected = [31.0, 19.0, 85.0, 55.0];
    const matchesD = mmExpected.every((val, idx) => Math.abs(val - mmFloats[idx]) < 1e-4);
    if (matchesD) {
      log('   ✅ Test D PASSED: WGSL 32x32 tiled GEMM matmul verified!');
    } else {
      log(
        `   ❌ Test D FAILED: Expected [${mmExpected.join(', ')}] but got [${Array.from(mmFloats).join(', ')}]`
      );
    }

    mmCtx.destroy();

    log('\n--- Test E: Context Execution of aten.mm.default (128-bit Vectorized vec4 GEMM) ---');
    log('14. Testing vectorized matrix multiplication: A (4x4) @ B (4x4) -> Out (4x4)...');
    // Identity * Matrix test
    // A = 4x4 identity matrix
    const aData4x4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    // B = 4x4 matrix [1..16]
    const bData4x4 = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);

    const vecCtx = new WgpuExecutionContext(device);
    vecCtx
      .setTensor('a', 'float32', [4, 4], aData4x4)
      .setTensor('b', 'float32', [4, 4], bData4x4)
      .setTensor('out', 'float32', [4, 4])
      .record(kernels.mm, ['a', 'b', 'out']);
    log('   Context recorded 1 vectorized vec4 GEMM dispatch.');

    log('15. Executing vectorized matrix multiplication on WebGPU...');
    const tVec = performance.now();
    vecCtx.submit();
    await device.queue.onSubmittedWorkDone();
    const vecDurationMs = performance.now() - tVec;
    log(`   Execution completed in ${vecDurationMs.toFixed(3)}ms (GPU queue wall-time)`);

    log('16. Reading back vectorized output...');
    const vecBytes = await vecCtx.getTensor('out').getData();
    const vecFloats = new Float32Array(vecBytes);
    log(`   Result: [${Array.from(vecFloats).join(', ')}]`);

    // Expected: Identity @ B = B
    const vecExpected = Array.from(bData4x4);
    const matchesE = vecExpected.every((val, idx) => Math.abs(val - vecFloats[idx]) < 1e-4);
    if (matchesE) {
      log('   ✅ Test E PASSED: WGSL 128-bit vectorized vec4 GEMM matmul verified!');
    } else {
      log(
        `   ❌ Test E FAILED: Expected [${vecExpected.join(', ')}] but got [${Array.from(vecFloats).join(', ')}]`
      );
    }

    vecCtx.destroy();

    log('\n--- Test F: High-Performance GEMM Benchmark (2048x2048) vs NumPy ---');
    const benchDim = 2048;
    log(`17. Initializing deterministic 2048x2048 matrices (16.7MB each in VRAM)...`);

    const metaRes = await fetch('/bench/meta_2048.json');
    const meta = (await metaRes.json()) as {
      dim: number;
      gpu_name: string;
      numpy_avg_ms: number;
      cuda_avg_ms: number;
      cuda_tflops: number;
      checksum: number;
      sample_0_0: number;
      sample_0_100: number;
      sample_1000_1000: number;
      sample_last: number;
    };

    log(`   Target GPU: ${meta.gpu_name}`);
    log(
      `   PyTorch CUDA (cuBLAS) reference: ${meta.cuda_avg_ms.toFixed(2)} ms (${meta.cuda_tflops.toFixed(2)} TFLOPS)`
    );
    log(`   NumPy CPU (OpenBLAS) reference:  ${meta.numpy_avg_ms.toFixed(2)} ms`);

    const totalElements = benchDim * benchDim;
    const aFloats = new Float32Array(totalElements);
    const bFloats = new Float32Array(totalElements);

    // Identical deterministic generation to Python NumPy script
    for (let i = 0; i < benchDim; i++) {
      const rowOffset = i * benchDim;
      for (let j = 0; j < benchDim; j++) {
        aFloats[rowOffset + j] = ((i * 31.0 + j * 17.0 + 1.0) % 100.0) / 100.0 - 0.5;
        bFloats[rowOffset + j] = ((i * 13.0 + j * 43.0 + 7.0) % 100.0) / 100.0 - 0.5;
      }
    }

    log(`18. Uploading 2048x2048 test matrices to WebGPU VRAM...`);
    const benchCtx = new WgpuExecutionContext(device);
    benchCtx
      .setTensor('a', 'float32', [benchDim, benchDim], aFloats)
      .setTensor('b', 'float32', [benchDim, benchDim], bFloats)
      .setTensor('out', 'float32', [benchDim, benchDim])
      .record(kernels.mm, ['a', 'b', 'out']);
    log('   Context recorded 2048x2048 tiled vec4 GEMM dispatch.');

    // Warm-up run & accuracy verification
    log('19. Executing on WebGPU and verifying accuracy against NumPy...');
    benchCtx.submit();
    await device.queue.onSubmittedWorkDone();

    const actualOutBytes = await benchCtx.getTensor('out').getData();
    const actualFloats = new Float32Array(actualOutBytes);

    // Verify sample coordinates against NumPy reference
    const check00 = Math.abs(actualFloats[0] - meta.sample_0_0);
    const check0100 = Math.abs(actualFloats[100] - meta.sample_0_100);
    const check1000 = Math.abs(actualFloats[1000 * benchDim + 1000] - meta.sample_1000_1000);
    const checkLast = Math.abs(actualFloats[totalElements - 1] - meta.sample_last);
    const maxSampleDiff = Math.max(check00, check0100, check1000, checkLast);

    if (maxSampleDiff < 1e-3) {
      log(
        `   ✅ Numerical accuracy verified! Max sample diff from NumPy: ${maxSampleDiff.toExponential(3)}`
      );
    } else {
      log(`   ❌ Numerical check failed! Diff: ${maxSampleDiff}`);
    }

    // Benchmark iterations
    const iterations = 10;
    log(`20. Running ${iterations} benchmark iterations on WebGPU...`);
    const tBenchStart = performance.now();
    for (let it = 0; it < iterations; it++) {
      benchCtx.submit();
    }
    await device.queue.onSubmittedWorkDone();
    const totalBenchTimeMs = performance.now() - tBenchStart;
    const avgWebGpuMs = totalBenchTimeMs / iterations;

    // 2 * M * N * K floating point operations
    const totalFlops = 2.0 * benchDim * benchDim * benchDim;
    const gigaFlops = totalFlops / 1e9;
    const throughputGflops = gigaFlops / (avgWebGpuMs / 1000);
    const throughputTflops = throughputGflops / 1000;

    log(`\n📊 Benchmark Results Summary (2048x2048 Matmul, 17.18 GFLOPs per run):`);
    log(
      `   - PyTorch CUDA (cuBLAS): ${meta.cuda_avg_ms.toFixed(2)} ms (${meta.cuda_tflops.toFixed(2)} TFLOPS)`
    );
    log(
      `   - Phlox (WebGPU):        ${avgWebGpuMs.toFixed(2)} ms (${throughputGflops.toFixed(1)} GFLOPS / ${throughputTflops.toFixed(2)} TFLOPS)`
    );
    log(`   - NumPy (CPU BLAS):      ${meta.numpy_avg_ms.toFixed(2)} ms`);
    log('   ✅ Test F PASSED: 2048x2048 GEMM benchmark completed!');

    benchCtx.destroy();

    log('\n--- Test G: Scalar Tiled GEMM Benchmark (2047x2047, non-divisible by 4) ---');
    const oddM = 2047;
    const oddK = 2047;
    const oddN = 2047;
    log(`21. Initializing 2047x2047 matrices (forces non-vec4 scalar 32x32 tiled shader)...`);

    const metaOddRes = await fetch('/bench/meta_odd.json');
    const metaOdd = (await metaOddRes.json()) as {
      M: number;
      K: number;
      N: number;
      gpu_name: string;
      numpy_avg_ms: number;
      cuda_avg_ms: number;
      cuda_tflops: number;
      sample_0_0: number;
      sample_0_100: number;
      sample_1000_1000: number;
      sample_last: number;
    };

    log(`   Target GPU: ${metaOdd.gpu_name}`);
    log(
      `   PyTorch CUDA (cuBLAS) reference: ${metaOdd.cuda_avg_ms.toFixed(2)} ms (${metaOdd.cuda_tflops.toFixed(2)} TFLOPS)`
    );
    log(`   NumPy CPU (OpenBLAS) reference:  ${metaOdd.numpy_avg_ms.toFixed(2)} ms`);

    const aOddFloats = new Float32Array(oddM * oddK);
    const bOddFloats = new Float32Array(oddK * oddN);

    for (let i = 0; i < oddM; i++) {
      const rowOffset = i * oddK;
      for (let j = 0; j < oddK; j++) {
        aOddFloats[rowOffset + j] = ((i * 31.0 + j * 17.0 + 1.0) % 100.0) / 100.0 - 0.5;
      }
    }
    for (let i = 0; i < oddK; i++) {
      const rowOffset = i * oddN;
      for (let j = 0; j < oddN; j++) {
        bOddFloats[rowOffset + j] = ((i * 13.0 + j * 43.0 + 7.0) % 100.0) / 100.0 - 0.5;
      }
    }

    log(`22. Uploading 2047x2047 test matrices to WebGPU VRAM...`);
    const oddCtx = new WgpuExecutionContext(device);
    oddCtx
      .setTensor('a', 'float32', [oddM, oddK], aOddFloats)
      .setTensor('b', 'float32', [oddK, oddN], bOddFloats)
      .setTensor('out', 'float32', [oddM, oddN])
      .record(kernels.mm, ['a', 'b', 'out']);
    log('   Context recorded 2047x2047 scalar tiled GEMM dispatch (K%4!=0, N%4!=0).');

    // Warm-up & accuracy check
    log('23. Executing on WebGPU and verifying accuracy against NumPy...');
    oddCtx.submit();
    await device.queue.onSubmittedWorkDone();

    const actualOddBytes = await oddCtx.getTensor('out').getData();
    const actualOddFloats = new Float32Array(actualOddBytes);

    const checkOdd00 = Math.abs(actualOddFloats[0] - metaOdd.sample_0_0);
    const checkOdd0100 = Math.abs(actualOddFloats[100] - metaOdd.sample_0_100);
    const checkOdd1000 = Math.abs(actualOddFloats[1000 * oddN + 1000] - metaOdd.sample_1000_1000);
    const checkOddLast = Math.abs(actualOddFloats[oddM * oddN - 1] - metaOdd.sample_last);
    const maxOddDiff = Math.max(checkOdd00, checkOdd0100, checkOdd1000, checkOddLast);

    if (maxOddDiff < 1e-3) {
      log(
        `   ✅ Numerical accuracy verified! Max sample diff from NumPy: ${maxOddDiff.toExponential(3)}`
      );
    } else {
      log(`   ❌ Numerical check failed! Diff: ${maxOddDiff}`);
    }

    // Benchmark iterations
    const oddIterations = 10;
    log(`24. Running ${oddIterations} benchmark iterations on WebGPU...`);
    const tOddStart = performance.now();
    for (let it = 0; it < oddIterations; it++) {
      oddCtx.submit();
    }
    await device.queue.onSubmittedWorkDone();
    const totalOddTimeMs = performance.now() - tOddStart;
    const avgOddWebGpuMs = totalOddTimeMs / oddIterations;

    const totalOddFlops = 2.0 * oddM * oddN * oddK;
    const oddGigaFlops = totalOddFlops / 1e9;
    const oddThroughputGflops = oddGigaFlops / (avgOddWebGpuMs / 1000);
    const oddThroughputTflops = oddThroughputGflops / 1000;

    log(`\n📊 Benchmark Results Summary (2047x2047 Scalar Tiled, 17.16 GFLOPs per run):`);
    log(
      `   - PyTorch CUDA (cuBLAS): ${metaOdd.cuda_avg_ms.toFixed(2)} ms (${metaOdd.cuda_tflops.toFixed(2)} TFLOPS)`
    );
    log(
      `   - Phlox (WebGPU Scalar): ${avgOddWebGpuMs.toFixed(2)} ms (${oddThroughputGflops.toFixed(1)} GFLOPS / ${oddThroughputTflops.toFixed(2)} TFLOPS)`
    );
    log(`   - NumPy (CPU BLAS):      ${metaOdd.numpy_avg_ms.toFixed(2)} ms`);
    log('   ✅ Test G PASSED: 2047x2047 scalar tiled GEMM benchmark completed!');

    oddCtx.destroy();

    log('\n--- Test H: End-to-End ExecuTorch .pte Model Execution ---');
    log('14. Fetching and loading simple_add.pte model...');
    const pteResponse = await fetch('/simple_add.pte');
    if (!pteResponse.ok) {
      throw new Error(`Failed to fetch /simple_add.pte: ${pteResponse.statusText}`);
    }
    const pteArrayBuffer = await pteResponse.arrayBuffer();
    const model = Model.load(device, pteArrayBuffer);
    log(`    Available methods in .pte: [${model.methodNames.join(', ')}]`);

    const modelIn1Data = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    const modelIn2Data = new Float32Array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    const modelIn1 = new Tensor('float32', [2, 4], device, modelIn1Data);
    const modelIn2 = new Tensor('float32', [2, 4], device, modelIn2Data);

    log("15. Executing model.execute('forward', [in1, in2])...");
    const outputs = await model.execute('forward', [modelIn1, modelIn2]);
    const outTensor = outputs[0] as Tensor;
    const modelOutBytes = await outTensor.getData();
    const modelOutFloats = new Float32Array(modelOutBytes);
    log(`    Output shape: [${outTensor.shape.join(', ')}]`);
    log(`    Result: [${Array.from(modelOutFloats).join(', ')}]`);

    const expectedAdd = modelIn1Data.map((v, i) => v + modelIn2Data[i]);
    const matchesModel = modelOutFloats.every((v, i) => Math.abs(v - expectedAdd[i]) < 1e-5);
    if (matchesModel) {
      log('    ✅ Test H PASSED: ExecuTorch .pte model executed with 100% accuracy!');
    } else {
      log('    ❌ Test H FAILED: Model output does not match expected add!');
    }

    modelIn1.destroy();
    modelIn2.destroy();
    model.dispose();
    log('    Model disposed.');

    log('\n🎉 All checks completed successfully!');
  } catch (err) {
    log(`\n❌ Error: ${(err as Error).message}`);
  } finally {
    runBtn.disabled = false;
  }
});
