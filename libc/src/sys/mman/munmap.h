//===-- Implementation header for mumap function ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_MMAN_MUNMAP_H
#define LLVM_LIBC_SRC_SYS_MMAN_MUNMAP_H

#include "include/sys/mman.h" // For size_t.

namespace __llvm_libc {

int munmap(void *addr, size_t size);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SYS_MMAN_MUNMAP_H
