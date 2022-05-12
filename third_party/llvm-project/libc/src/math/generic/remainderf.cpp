//===-- Implementation of remainderf function -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/remainderf.h"
#include "src/__support/FPUtil/DivisionAndRemainderOperations.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(float, remainderf, (float x, float y)) {
  int quotient;
  return fputil::remquo(x, y, quotient);
}

} // namespace __llvm_libc
