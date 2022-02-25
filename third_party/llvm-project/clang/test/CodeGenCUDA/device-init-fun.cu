// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device -std=c++11 \
// RUN:     -fgpu-allow-device-init -x hip \
// RUN:     -fno-threadsafe-statics -emit-llvm -o - %s \
// RUN:     | FileCheck %s

#include "Inputs/cuda.h"

// CHECK: define internal amdgpu_kernel void @_GLOBAL__sub_I_device_init_fun.cu() #[[ATTR:[0-9]*]]
// CHECK: attributes #[[ATTR]] = {{.*}}"device-init"

__device__ void f();

struct A {
  __device__ A() { f(); }
};

__device__ A a;
