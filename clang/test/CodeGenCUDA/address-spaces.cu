// RUN: %clang_cc1 -emit-llvm %s -o - -fcuda-is-device -triple ptx32-unknown-unknown | FileCheck %s

#include "../SemaCUDA/cuda.h"

// CHECK: @i = global
__device__ int i;

// CHECK: @j = addrspace(1) global
__constant__ int j;

// CHECK: @k = addrspace(4) global
__shared__ int k;

__device__ void foo() {
  // CHECK: load i32* @i
  i++;

  // CHECK: load i32* bitcast (i32 addrspace(1)* @j to i32*)
  j++;

  // CHECK: load i32* bitcast (i32 addrspace(4)* @k to i32*)
  k++;
}

