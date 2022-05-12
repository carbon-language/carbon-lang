//===-- Definition of mtx_t type ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_MTX_T_H__
#define __LLVM_LIBC_TYPES_MTX_T_H__

typedef struct {
  unsigned char __internal_data[4];
  int __mtx_type;
} mtx_t;

#endif // __LLVM_LIBC_TYPES_MTX_T_H__
