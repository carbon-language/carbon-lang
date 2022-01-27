//===-- Implementation header for _Exit -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC__EXIT_H
#define LLVM_LIBC_SRC__EXIT_H

namespace __llvm_libc {

[[noreturn]] void _Exit(int status);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC__EXIT_H
