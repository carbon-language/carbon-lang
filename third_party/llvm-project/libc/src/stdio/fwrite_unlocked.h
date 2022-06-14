//===-- Implementation header of fwrite_unlocked ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_FWRITE_UNLOCKED_H
#define LLVM_LIBC_SRC_STDIO_FWRITE_UNLOCKED_H

#include <stdio.h>

namespace __llvm_libc {

size_t fwrite_unlocked(const void *__restrict ptr, size_t size, size_t nmemb,
                       ::FILE *__restrict stream);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_FWRITE_UNLOCKED_H
