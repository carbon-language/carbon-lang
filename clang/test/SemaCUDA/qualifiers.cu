// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple nvptx-unknown-cuda -fsyntax-only -verify -fcuda-is-device %s

#include "Inputs/cuda.h"

// Host (x86) supports TLS and device-side compilation should ignore
// host variables. No errors in either case.
int __thread host_tls_var;

#if defined(__CUDA_ARCH__)
// NVPTX does not support TLS
__device__ int __thread device_tls_var; // expected-error {{thread-local storage is not supported for the current target}}
__shared__ int __thread shared_tls_var; // expected-error {{thread-local storage is not supported for the current target}}
#else
// Device-side vars should not produce any errors during host-side
// compilation.
__device__ int __thread device_tls_var;
__shared__ int __thread shared_tls_var;
#endif

__global__ void g1(int x) {}
__global__ int g2(int x) { // expected-error {{must have void return type}}
  return 1;
}
