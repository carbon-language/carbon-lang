//===-- Implementation of fedisableexcept function ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/fedisableexcept.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, fedisableexcept, (int e)) {
  return fputil::disable_except(e);
}

} // namespace __llvm_libc
