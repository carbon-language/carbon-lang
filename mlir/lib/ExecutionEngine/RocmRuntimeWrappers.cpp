//===- RocmRuntimeWrappers.cpp - MLIR ROCM runtime wrapper library --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C wrappers around the ROCM library for easy linking in ORC jit.
// Also adds some debugging helpers that are helpful when writing MLIR code to
// run on GPUs.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <numeric>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/ADT/ArrayRef.h"

#include "hip/hip_runtime.h"

#define HIP_REPORT_IF_ERROR(expr)                                              \
  [](hipError_t result) {                                                      \
    if (!result)                                                               \
      return;                                                                  \
    const char *name = hipGetErrorName(result);                                \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
  }(expr)

// Sets the `Context` for the duration of the instance and restores the previous
// context on destruction.
class ScopedContext {
public:
  ScopedContext() {
    // Static reference to HIP primary context for device ordinal 0.
    static hipCtx_t context = [] {
      HIP_REPORT_IF_ERROR(hipInit(/*flags=*/0));
      hipDevice_t device;
      HIP_REPORT_IF_ERROR(hipDeviceGet(&device, /*ordinal=*/0));
      hipCtx_t ctx;
      HIP_REPORT_IF_ERROR(hipDevicePrimaryCtxRetain(&ctx, device));
      return ctx;
    }();

    HIP_REPORT_IF_ERROR(hipCtxPushCurrent(context));
  }

  ~ScopedContext() { HIP_REPORT_IF_ERROR(hipCtxPopCurrent(nullptr)); }
};

extern "C" hipModule_t mgpuModuleLoad(void *data) {
  ScopedContext scopedContext;
  hipModule_t module = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleLoadData(&module, data));
  return module;
}

extern "C" void mgpuModuleUnload(hipModule_t module) {
  HIP_REPORT_IF_ERROR(hipModuleUnload(module));
}

extern "C" hipFunction_t mgpuModuleGetFunction(hipModule_t module,
                                               const char *name) {
  hipFunction_t function = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleGetFunction(&function, module, name));
  return function;
}

// The wrapper uses intptr_t instead of ROCM's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" void mgpuLaunchKernel(hipFunction_t function, intptr_t gridX,
                                 intptr_t gridY, intptr_t gridZ,
                                 intptr_t blockX, intptr_t blockY,
                                 intptr_t blockZ, int32_t smem,
                                 hipStream_t stream, void **params,
                                 void **extra) {
  ScopedContext scopedContext;
  HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ,
                                            blockX, blockY, blockZ, smem,
                                            stream, params, extra));
}

extern "C" hipStream_t mgpuStreamCreate() {
  ScopedContext scopedContext;
  hipStream_t stream = nullptr;
  HIP_REPORT_IF_ERROR(hipStreamCreate(&stream));
  return stream;
}

extern "C" void mgpuStreamDestroy(hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipStreamDestroy(stream));
}

extern "C" void mgpuStreamSynchronize(hipStream_t stream) {
  return HIP_REPORT_IF_ERROR(hipStreamSynchronize(stream));
}

extern "C" void mgpuStreamWaitEvent(hipStream_t stream, hipEvent_t event) {
  HIP_REPORT_IF_ERROR(hipStreamWaitEvent(stream, event, /*flags=*/0));
}

extern "C" hipEvent_t mgpuEventCreate() {
  ScopedContext scopedContext;
  hipEvent_t event = nullptr;
  HIP_REPORT_IF_ERROR(hipEventCreateWithFlags(&event, hipEventDisableTiming));
  return event;
}

extern "C" void mgpuEventDestroy(hipEvent_t event) {
  HIP_REPORT_IF_ERROR(hipEventDestroy(event));
}

extern "C" void mgpuEventSynchronize(hipEvent_t event) {
  HIP_REPORT_IF_ERROR(hipEventSynchronize(event));
}

extern "C" void mgpuEventRecord(hipEvent_t event, hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipEventRecord(event, stream));
}

extern "C" void *mgpuMemAlloc(uint64_t sizeBytes, hipStream_t /*stream*/) {
  ScopedContext scopedContext;
  void *ptr;
  HIP_REPORT_IF_ERROR(hipMalloc(&ptr, sizeBytes));
  return ptr;
}

extern "C" void mgpuMemFree(void *ptr, hipStream_t /*stream*/) {
  HIP_REPORT_IF_ERROR(hipFree(ptr));
}

extern "C" void mgpuMemcpy(void *dst, void *src, uint64_t sizeBytes,
                           hipStream_t stream) {
  HIP_REPORT_IF_ERROR(
      hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDefault, stream));
}

/// Helper functions for writing mlir example code

// Allows to register byte array with the ROCM runtime. Helpful until we have
// transfer functions implemented.
extern "C" void mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  ScopedContext scopedContext;
  HIP_REPORT_IF_ERROR(hipHostRegister(ptr, sizeBytes, /*flags=*/0));
}

// Allows to register a MemRef with the ROCm runtime. Helpful until we have
// transfer functions implemented.
extern "C" void
mgpuMemHostRegisterMemRef(int64_t rank, StridedMemRefType<char, 1> *descriptor,
                          int64_t elementSizeBytes) {

  llvm::SmallVector<int64_t, 4> denseStrides(rank);
  llvm::ArrayRef<int64_t> sizes(descriptor->sizes, rank);
  llvm::ArrayRef<int64_t> strides(sizes.end(), rank);

  std::partial_sum(sizes.rbegin(), sizes.rend(), denseStrides.rbegin(),
                   std::multiplies<int64_t>());
  auto sizeBytes = denseStrides.front() * elementSizeBytes;

  // Only densely packed tensors are currently supported.
  std::rotate(denseStrides.begin(), denseStrides.begin() + 1,
              denseStrides.end());
  denseStrides.back() = 1;
  assert(strides == llvm::makeArrayRef(denseStrides));

  auto ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostRegister(ptr, sizeBytes);
}

template <typename T>
void mgpuMemGetDevicePointer(T *hostPtr, T **devicePtr) {
  HIP_REPORT_IF_ERROR(hipSetDevice(0));
  HIP_REPORT_IF_ERROR(
      hipHostGetDevicePointer((void **)devicePtr, hostPtr, /*flags=*/0));
}

extern "C" StridedMemRefType<float, 1>
mgpuMemGetDeviceMemRef1dFloat(float *allocated, float *aligned, int64_t offset,
                              int64_t size, int64_t stride) {
  float *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

extern "C" StridedMemRefType<int32_t, 1>
mgpuMemGetDeviceMemRef1dInt32(int32_t *allocated, int32_t *aligned,
                              int64_t offset, int64_t size, int64_t stride) {
  int32_t *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}
