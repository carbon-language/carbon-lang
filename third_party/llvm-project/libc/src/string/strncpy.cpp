//===-- Implementation of strncpy -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strncpy.h"

#include "src/__support/common.h"
#include <stddef.h> // For size_t.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(char *, strncpy,
                   (char *__restrict dest, const char *__restrict src,
                    size_t n)) {
  size_t i = 0;
  // Copy up until \0 is found.
  for (; i < n && src[i] != '\0'; ++i)
    dest[i] = src[i];
  // When n>strlen(src), n-strlen(src) \0 are appended.
  for (; i < n; ++i)
    dest[i] = '\0';
  return dest;
}

} // namespace __llvm_libc
