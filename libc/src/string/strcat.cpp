//===-- Implementation of strcat ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcat.h"
#include "src/string/strcpy.h"
#include "src/string/string_utils.h"

#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(char *, strcat,
                   (char *__restrict dest, const char *__restrict src)) {
  __llvm_libc::strcpy(dest + internal::string_length(dest), src);
  return dest;
}

} // namespace __llvm_libc
