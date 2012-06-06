//===-- asan_printf.cc ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Internal printf function, used inside ASan run-time library.
// We can't use libc printf because we intercept some of the functions used
// inside it.
//===----------------------------------------------------------------------===//

#include "asan_internal.h"
#include "asan_interceptors.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_common.h"

#include <stdarg.h>
#include <stdio.h>

namespace __sanitizer {
int VSNPrintf(char *buff, int buff_length, const char *format, va_list args);
}  // namespace __sanitizer

namespace __asan {

void AsanPrintf(const char *format, ...) {
  const int kLen = 1024 * 4;
  char buffer[kLen];
  va_list args;
  va_start(args, format);
  int needed_length = VSNPrintf(buffer, kLen, format, args);
  va_end(args);
  RAW_CHECK_MSG(needed_length < kLen, "Buffer in Printf is too short!\n");
  RawWrite(buffer);
  AppendToErrorMessageBuffer(buffer);
}

// Like AsanPrintf, but prints the current PID before the output string.
void AsanReport(const char *format, ...) {
  const int kLen = 1024 * 4;
  char buffer[kLen];
  int needed_length = SNPrintf(buffer, kLen, "==%d== ", GetPid());
  RAW_CHECK_MSG(needed_length < kLen, "Buffer in Report is too short!\n");
  va_list args;
  va_start(args, format);
  needed_length += VSNPrintf(buffer + needed_length, kLen - needed_length,
                             format, args);
  va_end(args);
  RAW_CHECK_MSG(needed_length < kLen, "Buffer in Report is too short!\n");
  RawWrite(buffer);
  AppendToErrorMessageBuffer(buffer);
}

}  // namespace __asan
