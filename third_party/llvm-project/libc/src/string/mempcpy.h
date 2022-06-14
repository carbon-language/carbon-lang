//===-- Implementation header for mempcpy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMPCPY_H
#define LLVM_LIBC_SRC_STRING_MEMPCPY_H

#include <stddef.h>

namespace __llvm_libc {

void *mempcpy(void *__restrict dest, const void *__restrict src, size_t count);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMPCPY_H
