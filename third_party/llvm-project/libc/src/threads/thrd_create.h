//===-- Implementation header for thrd_create function ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_THREADS_THRD_CREATE_H
#define LLVM_LIBC_SRC_THREADS_THRD_CREATE_H

#include <threads.h>

namespace __llvm_libc {

int thrd_create(thrd_t *thread, thrd_start_t func, void *arg);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_THREADS_THRD_CREATE_H
