// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -fsyntax-only \
// RUN:   -verify -DEXPECT_VA_ARG_ERR -DEXPECT_VARARG_ERR %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -fsyntax-only \
// RUN:   -fcuda-allow-variadic-functions -verify -DEXPECT_VA_ARG_ERR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify \
// RUN:   -DEXPECT_VARARG_ERR %s

#include <stdarg.h>
#include "Inputs/cuda.h"

__device__ void foo() {
  va_list list;
  va_arg(list, int);
#ifdef EXPECT_VA_ARG_ERR
  // expected-error@-2 {{CUDA device code does not support va_arg}}
#endif
}

void bar() {
  va_list list;
  va_arg(list, int);  // OK: host-only
}

__device__ void baz() {
#if !defined(__CUDA_ARCH__)
  va_list list;
  va_arg(list, int);  // OK: only seen when compiling for host
#endif
}

__device__ void vararg(const char* x, ...) {}
#ifdef EXPECT_VARARG_ERR
// expected-error@-2 {{CUDA device code does not support variadic functions}}
#endif

template <typename T>
__device__ void vararg(T t, ...) {}
#ifdef EXPECT_VARARG_ERR
// expected-error@-2 {{CUDA device code does not support variadic functions}}
#endif

extern "C" __device__ int printf(const char* fmt, ...);  // OK, special case.

// Definition of printf not allowed.
extern "C" __device__ int printf(const char* fmt, ...) { return 0; }
#ifdef EXPECT_VARARG_ERR
// expected-error@-2 {{CUDA device code does not support variadic functions}}
#endif

namespace ns {
__device__ int printf(const char* fmt, ...);
#ifdef EXPECT_VARARG_ERR
// expected-error@-2 {{CUDA device code does not support variadic functions}}
#endif
}
