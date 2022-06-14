//===-- x86_64 implementations of the fma function --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_FMA_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_FMA_H

#include "src/__support/architectures.h"

#if !defined(LLVM_LIBC_ARCH_X86_64)
#error "Invalid include"
#endif

#if !defined(LIBC_TARGET_HAS_FMA)
#error "FMA instructions are not supported"
#endif

#include "src/__support/CPP/TypeTraits.h"
#include <immintrin.h>

namespace __llvm_libc {
namespace fputil {

template <typename T>
static inline cpp::EnableIfType<cpp::IsSame<T, float>::Value, T> fma(T x, T y,
                                                                     T z) {
  float result;
  __m128 xmm = _mm_load_ss(&x);           // NOLINT
  __m128 ymm = _mm_load_ss(&y);           // NOLINT
  __m128 zmm = _mm_load_ss(&z);           // NOLINT
  __m128 r = _mm_fmadd_ss(xmm, ymm, zmm); // NOLINT
  _mm_store_ss(&result, r);               // NOLINT
  return result;
}

template <typename T>
static inline cpp::EnableIfType<cpp::IsSame<T, double>::Value, T> fma(T x, T y,
                                                                      T z) {
  double result;
  __m128d xmm = _mm_load_sd(&x);           // NOLINT
  __m128d ymm = _mm_load_sd(&y);           // NOLINT
  __m128d zmm = _mm_load_sd(&z);           // NOLINT
  __m128d r = _mm_fmadd_sd(xmm, ymm, zmm); // NOLINT
  _mm_store_sd(&result, r);                // NOLINT
  return result;
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_FMA_H
