//===-- Implementation header of fread_unlocked -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_FREAD_UNLOCKED_H
#define LLVM_LIBC_SRC_STDIO_FREAD_UNLOCKED_H

#include <stdio.h>

namespace __llvm_libc {

size_t fread_unlocked(void *__restrict buffer, size_t size, size_t nmemb,
                      ::FILE *stream);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_FREAD_UNLOCKED_H
