//===-- Implementation of asctime_r function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/asctime_r.h"
#include "src/__support/common.h"
#include "src/time/time_utils.h"

namespace __llvm_libc {

using __llvm_libc::time_utils::TimeConstants;

LLVM_LIBC_FUNCTION(char *, asctime_r,
                   (const struct tm *timeptr, char *buffer)) {
  return time_utils::asctime(timeptr, buffer, TimeConstants::ASCTIME_MAX_BYTES);
}

} // namespace __llvm_libc
