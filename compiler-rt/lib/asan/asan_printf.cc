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

#include <stdarg.h>
#include <stdio.h>

using namespace __sanitizer;  // NOLINT

namespace __asan {

extern char *error_message_buffer;
extern uptr error_message_buffer_pos, error_message_buffer_size;

void RawWrite(const char *buffer) {
  static const char *kRawWriteError = "RawWrite can't output requested buffer!";
  uptr length = (uptr)internal_strlen(buffer);
  if (length != AsanWrite(2, buffer, length)) {
    AsanWrite(2, kRawWriteError, internal_strlen(kRawWriteError));
    AsanDie();
  }
  if (error_message_buffer) {
    int remaining = error_message_buffer_size - error_message_buffer_pos;
    internal_strncpy(error_message_buffer + error_message_buffer_pos,
                     buffer, remaining);
    error_message_buffer[error_message_buffer_size - 1] = '\0';
    // FIXME: reallocate the buffer instead of truncating the message.
    error_message_buffer_pos += remaining > length ? length : remaining;
  }
}

static inline int AppendChar(char **buff, const char *buff_end, char c) {
  if (*buff < buff_end) {
    **buff = c;
    (*buff)++;
  }
  return 1;
}

// Appends number in a given base to buffer. If its length is less than
// "minimal_num_length", it is padded with leading zeroes.
static int AppendUnsigned(char **buff, const char *buff_end, u64 num,
                          u8 base, u8 minimal_num_length) {
  uptr const kMaxLen = 30;
  RAW_CHECK(base == 10 || base == 16);
  RAW_CHECK(minimal_num_length < kMaxLen);
  uptr num_buffer[kMaxLen];
  uptr pos = 0;
  do {
    RAW_CHECK_MSG(pos < kMaxLen, "appendNumber buffer overflow");
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

static inline int AppendSignedDecimal(char **buff, const char *buff_end,
                                      s64 num) {
  int result = 0;
  if (num < 0) {
    result += AppendChar(buff, buff_end, '-');
    num = -num;
  }
  result += AppendUnsigned(buff, buff_end, (u64)num, 10, 0);
  return result;
}

static inline int AppendString(char **buff, const char *buff_end,
                               const char *s) {
  // Avoid library functions like stpcpy here.
  RAW_CHECK_MSG(s, "Error: passing a 0 pointer to AppendString\n");
  int result = 0;
  for (; *s; s++) {
    result += AppendChar(buff, buff_end, *s);
  }
  return result;
}

static inline int AppendPointer(char **buff, const char *buff_end,
                                u64 ptr_value) {
  int result = 0;
  result += AppendString(buff, buff_end, "0x");
  result += AppendUnsigned(buff, buff_end, ptr_value, 16,
                           (__WORDSIZE == 64) ? 12 : 8);
  return result;
}

static int VSNPrintf(char *buff, int buff_length,
                     const char *format, va_list args) {
  static const char *kPrintfFormatsHelp = "Supported Printf formats: "
                                          "%%[z]{d,u,x}; %%p; %%s";
  RAW_CHECK(format);
  RAW_CHECK(buff_length > 0);
  const char *buff_end = &buff[buff_length - 1];
  const char *cur = format;
  int result = 0;
  for (; *cur; cur++) {
    if (*cur == '%') {
      cur++;
      bool have_z = (*cur == 'z');
      cur += have_z;
      s64 dval;
      u64 uval;
      switch (*cur) {
        case 'd': dval = have_z ? va_arg(args, sptr)
                                : va_arg(args, int);
                  result += AppendSignedDecimal(&buff, buff_end, dval);
                  break;
        case 'u': uval = have_z ? va_arg(args, uptr)
                                : va_arg(args, unsigned);
                  result += AppendUnsigned(&buff, buff_end, uval, 10, 0);
                  break;
        case 'x': uval = have_z ? va_arg(args, uptr)
                                : va_arg(args, unsigned);
                  result += AppendUnsigned(&buff, buff_end, uval, 16, 0);
                  break;
        case 'p': RAW_CHECK_MSG(!have_z, kPrintfFormatsHelp);
                  result += AppendPointer(&buff, buff_end,
                                          va_arg(args, uptr));
                  break;
        case 's': RAW_CHECK_MSG(!have_z, kPrintfFormatsHelp);
                  result += AppendString(&buff, buff_end, va_arg(args, char*));
                  break;
        default:  RAW_CHECK_MSG(false, kPrintfFormatsHelp);
      }
    } else {
      result += AppendChar(&buff, buff_end, *cur);
    }
  }
  RAW_CHECK(buff <= buff_end);
  AppendChar(&buff, buff_end + 1, '\0');
  return result;
}

void Printf(const char *format, ...) {
  const int kLen = 1024 * 4;
  char buffer[kLen];
  va_list args;
  va_start(args, format);
  int needed_length = VSNPrintf(buffer, kLen, format, args);
  va_end(args);
  RAW_CHECK_MSG(needed_length < kLen, "Buffer in Printf is too short!\n");
  RawWrite(buffer);
}

// Writes at most "length" symbols to "buffer" (including trailing '\0').
// Returns the number of symbols that should have been written to buffer
// (not including trailing '\0'). Thus, the string is truncated
// iff return value is not less than "length".
int SNPrintf(char *buffer, uptr length, const char *format, ...) {
  va_list args;
  va_start(args, format);
  int needed_length = VSNPrintf(buffer, length, format, args);
  va_end(args);
  return needed_length;
}

// Like Printf, but prints the current PID before the output string.
void Report(const char *format, ...) {
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
}

int SScanf(const char *str, const char *format, ...) {
#ifndef _WIN32
  va_list args;
  va_start(args, format);
  int res = vsscanf(str, format, args);
  va_end(args);
  return res;
#else
  UNIMPLEMENTED();
  return -1;
#endif
}

}  // namespace __asan
