//===-- Definition of pthread_t type --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_PTHREAD_T_H__
#define __LLVM_LIBC_TYPES_PTHREAD_T_H__

#include <llvm-libc-types/__thread_type.h>

typedef __thread_type pthread_t;

#endif // __LLVM_LIBC_TYPES_PTHREAD_T_H__
