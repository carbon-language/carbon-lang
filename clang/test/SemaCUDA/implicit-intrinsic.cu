// RUN: %clang_cc1 -std=gnu++11 -triple nvptx64-unknown-unknown -fsyntax-only -verify %s

#include "Inputs/cuda.h"

// expected-no-diagnostics
__device__ void __threadfence_system() {
  // This shouldn't produce an error, since __nvvm_membar_sys is inferred to
  // be __host__ __device__ and thus callable from device code.
  __nvvm_membar_sys();
}
