//===-- Implementation of strtoimax ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/inttypes/strtoimax.h"
#include "src/__support/common.h"
#include "src/__support/str_conv_utils.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(intmax_t, strtoimax,
                   (const char *__restrict str, char **__restrict str_end,
                    int base)) {
  return internal::strtointeger<intmax_t>(str, str_end, base);
}

} // namespace __llvm_libc
