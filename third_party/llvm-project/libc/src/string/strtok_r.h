//===-- Implementation header for strtok_r ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_STRTOK_R_H
#define LLVM_LIBC_SRC_STRING_STRTOK_R_H

namespace __llvm_libc {

char *strtok_r(char *__restrict src, const char *__restrict delimiter_string,
               char **__restrict saveptr);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_STRTOK_R_H
