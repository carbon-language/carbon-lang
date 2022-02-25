//===-- Implementation of imaxdiv -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/inttypes/imaxdiv.h"
#include "src/__support/common.h"
#include "src/__support/integer_operations.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(imaxdiv_t, imaxdiv, (intmax_t x, intmax_t y)) {
  imaxdiv_t res;
  integer_rem_quo(x, y, res.quot, res.rem);
  return res;
}

} // namespace __llvm_libc
