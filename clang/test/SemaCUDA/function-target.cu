// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -fcuda-is-device -verify %s

#include "Inputs/cuda.h"

__host__ void h1h(void);
__device__ void h1d(void); // expected-note {{candidate function not viable: call to __device__ function from __host__ function}}
__host__ __device__ void h1hd(void);
__global__ void h1g(void);

struct h1ds { // expected-note {{requires 1 argument}}
  __device__ h1ds(); // expected-note {{candidate constructor not viable: call to __device__ function from __host__ function}}
};

__host__ void h1(void) {
  h1h();
  h1d(); // expected-error {{no matching function}}
  h1hd();
  h1g<<<1, 1>>>();
  h1ds x; // expected-error {{no matching constructor}}
}

__host__ void d1h(void); // expected-note {{candidate function not viable: call to __host__ function from __device__ function}}
__device__ void d1d(void);
__host__ __device__ void d1hd(void);
__global__ void d1g(void); // expected-note {{'d1g' declared here}}

__device__ void d1(void) {
  d1h(); // expected-error {{no matching function}}
  d1d();
  d1hd();
  d1g<<<1, 1>>>(); // expected-error {{reference to __global__ function 'd1g' in __device__ function}}
}

// Expected 0-1 as in one of host/device side compilation it is an error, while
// not in the other
__host__ void hd1h(void); // expected-note 0-1 {{candidate function not viable: call to __host__ function from __host__ __device__ function}}
__device__ void hd1d(void); // expected-note 0-1 {{candidate function not viable: call to __device__ function from __host__ __device__ function}}
__host__ void hd1hg(void);
__device__ void hd1dg(void);
#ifdef __CUDA_ARCH__
__host__ void hd1hig(void); // expected-note {{candidate function not viable: call to __host__ function from __host__ __device__ function}}
#else
__device__ void hd1dig(void); // expected-note {{candidate function not viable: call to __device__ function from __host__ __device__ function}}
#endif
__host__ __device__ void hd1hd(void);
__global__ void hd1g(void); // expected-note {{'hd1g' declared here}}

__host__ __device__ void hd1(void) {
  // Expected 0-1 as in one of host/device side compilation it is an error,
  // while not in the other
  hd1d(); // expected-error 0-1 {{no matching function}}
  hd1h(); // expected-error 0-1 {{no matching function}}

  // No errors as guarded
#ifdef __CUDA_ARCH__
  hd1d();
#else
  hd1h();
#endif

  // Errors as incorrectly guarded
#ifndef __CUDA_ARCH__
  hd1dig(); // expected-error {{no matching function}}
#else
  hd1hig(); // expected-error {{no matching function}}
#endif

  hd1hd();
  hd1g<<<1, 1>>>(); // expected-error {{reference to __global__ function 'hd1g' in __host__ __device__ function}}
}
