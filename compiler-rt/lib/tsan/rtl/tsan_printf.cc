//===-- tsan_printf.cc ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_libc.h"
#include "tsan_defs.h"
#include "tsan_mman.h"
#include "tsan_platform.h"

#include <stdarg.h>  // va_list

namespace __sanitizer {
int VSNPrintf(char *buff, int buff_length, const char *format, va_list args);
}  // namespace __sanitizer

namespace __tsan {

void TsanPrintf(const char *format, ...) {
  ScopedInRtl in_rtl;
  const uptr kMaxLen = 16 * 1024;
  InternalScopedBuf<char> buffer(kMaxLen);
  va_list args;
  va_start(args, format);
  uptr len = VSNPrintf(buffer, buffer.Size(), format, args);
  va_end(args);
  internal_write(CTX() ? flags()->log_fileno : 2,
      buffer, len < buffer.Size() ? len : buffer.Size() - 1);
}

}  // namespace __tsan
