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

#include "tsan_defs.h"
#include "tsan_mman.h"
#include "tsan_platform.h"

#include <stdarg.h>  // va_list

typedef long long i64;  // NOLINT
typedef long iptr;  // NOLINT

namespace __tsan {

static int AppendChar(char **buff, const char *buff_end, char c) {
  if (*buff < buff_end) {
    **buff = c;
    (*buff)++;
  }
  return 1;
}

static int AppendUnsigned(char **buff, const char *buff_end, u64 num,
                          int base, uptr minimal_num_length) {
  uptr const kMaxLen = 30;
  uptr num_buffer[kMaxLen];
  uptr pos = 0;
  do {
    num_buffer[pos++] = num % base;
    num /= base;
  } while (num > 0);
  while (pos < minimal_num_length) num_buffer[pos++] = 0;
  int result = 0;
  while (pos-- > 0) {
    uptr digit = num_buffer[pos];
    result += AppendChar(buff, buff_end, (digit < 10) ? '0' + digit
                                                      : 'a' + digit - 10);
  }
  return result;
}

static int AppendSignedDecimal(char **buff, const char *buff_end, i64 num) {
  int result = 0;
  if (num < 0) {
    result += AppendChar(buff, buff_end, '-');
    num = -num;
  }
  result += AppendUnsigned(buff, buff_end, (u64)num, 10, 0);
  return result;
}

static int AppendString(char **buff, const char *buff_end, const char *s) {
  if (s == 0)
    s = "<null>";
  int result = 0;
  for (; *s; s++) {
    result += AppendChar(buff, buff_end, *s);
  }
  return result;
}

static int AppendPointer(char **buff, const char *buff_end, u64 ptr_value) {
  int result = 0;
  result += AppendString(buff, buff_end, "0x");
  result += AppendUnsigned(buff, buff_end, ptr_value, 16,
      (sizeof(void*) == 8) ? 12 : 8);  // NOLINT
  return result;
}

static uptr VSNPrintf(char *buff, int buff_length,
                     const char *format, va_list args) {
  const char *buff_end = &buff[buff_length - 1];
  const char *cur = format;
  int result = 0;
  for (; *cur; cur++) {
    if (*cur != '%') {
      result += AppendChar(&buff, buff_end, *cur);
      continue;
    }
    cur++;
    bool is_long = (*cur == 'l');
    cur += is_long;
    bool is_llong = (*cur == 'l');
    cur += is_llong;
    switch (*cur) {
      case 'd': {
        i64 v = is_llong ? va_arg(args, i64)
            : is_long ? va_arg(args, iptr)
            : va_arg(args, int);
        result += AppendSignedDecimal(&buff, buff_end, v);
        break;
      }
      case 'u':
      case 'x': {
        u64 v = is_llong ? va_arg(args, u64)
            : is_long ? va_arg(args, uptr)
            : va_arg(args, unsigned);
        result += AppendUnsigned(&buff, buff_end, v, *cur == 'u' ? 10: 16, 0);
        break;
      }
      case 'p': {
        result += AppendPointer(&buff, buff_end, va_arg(args, uptr));
        break;
      }
      case 's': {
        result += AppendString(&buff, buff_end, va_arg(args, char*));
        break;
      }
      default: {
        Die();
      }
    }
  }
  AppendChar(&buff, buff_end + 1, '\0');
  return result;
}

void Printf(const char *format, ...) {
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

uptr Snprintf(char *buffer, uptr length, const char *format, ...) {
  va_list args;
  va_start(args, format);
  uptr len = VSNPrintf(buffer, length, format, args);
  va_end(args);
  return len;
}
}
