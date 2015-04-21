// RUN: %clang_cc1 "-triple" "nvptx-nvidia-cuda" -emit-llvm -fcuda-is-device -o - %s | FileCheck %s

#include "cuda_builtin_vars.h"

// CHECK: define void @_Z6kernelPi(i32* %out)
__attribute__((global))
void kernel(int *out) {
  int i = 0;
  out[i++] = threadIdx.x; // CHECK: call i32 @llvm.ptx.read.tid.x()
  out[i++] = threadIdx.y; // CHECK: call i32 @llvm.ptx.read.tid.y()
  out[i++] = threadIdx.z; // CHECK: call i32 @llvm.ptx.read.tid.z()

  out[i++] = blockIdx.x; // CHECK: call i32 @llvm.ptx.read.ctaid.x()
  out[i++] = blockIdx.y; // CHECK: call i32 @llvm.ptx.read.ctaid.y()
  out[i++] = blockIdx.z; // CHECK: call i32 @llvm.ptx.read.ctaid.z()

  out[i++] = blockDim.x; // CHECK: call i32 @llvm.ptx.read.ntid.x()
  out[i++] = blockDim.y; // CHECK: call i32 @llvm.ptx.read.ntid.y()
  out[i++] = blockDim.z; // CHECK: call i32 @llvm.ptx.read.ntid.z()

  out[i++] = gridDim.x; // CHECK: call i32 @llvm.ptx.read.nctaid.x()
  out[i++] = gridDim.y; // CHECK: call i32 @llvm.ptx.read.nctaid.y()
  out[i++] = gridDim.z; // CHECK: call i32 @llvm.ptx.read.nctaid.z()

  out[i++] = warpSize; // CHECK: store i32 32,

  // CHECK: ret void
}
