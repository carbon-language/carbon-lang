//===-- sanitizer_posix.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements POSIX-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//
#if defined(__linux__) || defined(__APPLE__)

#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"

#include <stdarg.h>
#include <stdio.h>

namespace __sanitizer {

int internal_sscanf(const char *str, const char *format, ...) {
  va_list args;
  va_start(args, format);
  int res = vsscanf(str, format, args);
  va_end(args);
  return res;
}

}  // namespace __sanitizer

#endif  // __linux__ || __APPLE_
