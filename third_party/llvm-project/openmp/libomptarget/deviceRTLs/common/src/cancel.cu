//===------ cancel.cu - NVPTX OpenMP cancel interface ------------ CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface to be used in the implementation of OpenMP cancel.
//
//===----------------------------------------------------------------------===//
#pragma omp declare target

#include "common/debug.h"
#include "interface.h"

EXTERN int32_t __kmpc_cancellationpoint(kmp_Ident *loc, int32_t global_tid,
                                        int32_t cancelVal) {
  PRINT(LD_IO, "call kmpc_cancellationpoint(cancel val %d)\n", (int)cancelVal);
  // disabled
  return 0;
}

EXTERN int32_t __kmpc_cancel(kmp_Ident *loc, int32_t global_tid,
                             int32_t cancelVal) {
  PRINT(LD_IO, "call kmpc_cancel(cancel val %d)\n", (int)cancelVal);
  // disabled
  return 0;
}

#pragma omp end declare target
