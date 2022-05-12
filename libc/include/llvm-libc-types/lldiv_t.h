//===-- Definition of type lldiv_t ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_LLDIV_T_H__
#define __LLVM_LIBC_TYPES_LLDIV_T_H__

typedef struct {
  long long quot;
  long long rem;
} lldiv_t;

#endif // __LLVM_LIBC_TYPES_LLDIV_T_H__
