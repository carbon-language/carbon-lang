//===-- Implementation header for pthread_attr_setguardsize -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_ATTR_SETGUARDSIZE_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_ATTR_SETGUARDSIZE_H

#include <pthread.h>

namespace __llvm_libc {

int pthread_attr_setguardsize(pthread_attr_t *attr, size_t guardsize);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_ATTR_SETGUARDSIZE_H
