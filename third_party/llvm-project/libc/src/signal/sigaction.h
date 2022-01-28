//===-- Implementation header for sigaction ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SIGNAL_SIGACTION_H
#define LLVM_LIBC_SRC_SIGNAL_SIGACTION_H

#define __LLVM_LIBC_INTERNAL_SIGACTION
#include "include/signal.h"

namespace __llvm_libc {

int sigaction(int signal, const struct __sigaction *__restrict libc_new,
              struct __sigaction *__restrict libc_old);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SIGNAL_SIGACTION_H
