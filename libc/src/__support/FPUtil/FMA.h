//===-- Common header for FMA implementations -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_FMA_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_FMA_H

#include "src/__support/architectures.h"

#if defined(LLVM_LIBC_ARCH_X86_64)
#include "x86_64/FMA.h"
#elif defined(LLVM_LIBC_ARCH_AARCH64)
#include "aarch64/FMA.h"
#else
#include "generic/FMA.h"
#include "src/__support/CPP/TypeTraits.h"

namespace __llvm_libc {
namespace fputil {

// We have a generic implementation available only for single precision fma as
// we restrict it to float values for now.
template <typename T>
static inline cpp::EnableIfType<cpp::IsSame<T, float>::Value, T> fma(T x, T y,
                                                                     T z) {
  return generic::fma(x, y, z);
}

} // namespace fputil
} // namespace __llvm_libc

#endif

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_FMA_H
