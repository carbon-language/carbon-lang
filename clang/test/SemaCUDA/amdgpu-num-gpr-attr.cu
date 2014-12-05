// RUN: %clang_cc1 -fsyntax-only -verify %s

#include "Inputs/cuda.h"

__attribute__((amdgpu_num_vgpr(64)))
__global__ void test_num_vgpr() { } // expected-error {{'amdgpu_num_vgpr' attribute only applies to kernel functions}}

__attribute__((amdgpu_num_sgpr(32)))
__global__ void test_num_sgpr() { } // expected-error {{'amdgpu_num_sgpr' attribute only applies to kernel functions}}

// fixme-expected-error@+3 {{'amdgpu_num_sgpr' attribute only applies to kernel functions}}
// expected-error@+2 {{'amdgpu_num_vgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_num_sgpr(32), amdgpu_num_vgpr(64)))
__global__ void test_num_vgpr_num_sgpr() { }
