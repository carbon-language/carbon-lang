//===-- Implementation header for modff -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_MODFF_H
#define LLVM_LIBC_SRC_MATH_MODFF_H

namespace __llvm_libc {

float modff(float x, float *iptr);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_MODFF_H
