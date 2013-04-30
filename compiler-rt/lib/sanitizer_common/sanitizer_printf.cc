//===-- sanitizer_printf.cc -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer.
//
// Internal printf function, used inside run-time libraries.
// We can't use libc printf because we intercept some of the functions used
// inside it.
//===----------------------------------------------------------------------===//


#include "sanitizer_common.h"
#include "sanitizer_libc.h"

#include <stdio.h>
#include <stdarg.h>

namespace __sanitizer {

StaticSpinMutex CommonSanitizerReportMutex;

static int AppendChar(char **buff, const char *buff_end, char c) {
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
  if (pos < minimal_num_length) {
    // Make sure compiler doesn't insert call to memset here.
    internal_memset(&num_buffer[pos], 0,
                    sizeof(num_buffer[0]) * (minimal_num_length - pos));
    pos = minimal_num_length;
  }
  int result = 0;
  while (pos-- > 0) {
    uptr digit = num_buffer[pos];
    result += AppendChar(buff, buff_end, (digit < 10) ? '0' + digit
                                                      : 'a' + digit - 10);
  }
  return result;
}

static int AppendSignedDecimal(char **buff, const char *buff_end, s64 num,
                               u8 minimal_num_length) {
  int result = 0;
  if (num < 0) {
    result += AppendChar(buff, buff_end, '-');
    num = -num;
    if (minimal_num_length)
      --minimal_num_length;
  }
  result += AppendUnsigned(buff, buff_end, (u64)num, 10, minimal_num_length);
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
                           (SANITIZER_WORDSIZE == 64) ? 12 : 8);
  return result;
}

int VSNPrintf(char *buff, int buff_length,
              const char *format, va_list args) {
  static const char *kPrintfFormatsHelp =
    "Supported Printf formats: %(0[0-9]*)?(z|ll)?{d,u,x}; %p; %s; %c\n";
  RAW_CHECK(format);
  RAW_CHECK(buff_length > 0);
  const char *buff_end = &buff[buff_length - 1];
  const char *cur = format;
  int result = 0;
  for (; *cur; cur++) {
    if (*cur != '%') {
      result += AppendChar(&buff, buff_end, *cur);
      continue;
    }
    cur++;
    bool have_width = (*cur == '0');
    int width = 0;
    if (have_width) {
      while (*cur >= '0' && *cur <= '9') {
        have_width = true;
        width = width * 10 + *cur++ - '0';
      }
    }
    bool have_z = (*cur == 'z');
    cur += have_z;
    bool have_ll = !have_z && (cur[0] == 'l' && cur[1] == 'l');
    cur += have_ll * 2;
    s64 dval;
    u64 uval;
    bool have_flags = have_width | have_z | have_ll;
    switch (*cur) {
      case 'd': {
        dval = have_ll ? va_arg(args, s64)
             : have_z ? va_arg(args, sptr)
             : va_arg(args, int);
        result += AppendSignedDecimal(&buff, buff_end, dval, width);
        break;
      }
      case 'u':
      case 'x': {
        uval = have_ll ? va_arg(args, u64)
             : have_z ? va_arg(args, uptr)
             : va_arg(args, unsigned);
        result += AppendUnsigned(&buff, buff_end, uval,
                                 (*cur == 'u') ? 10 : 16, width);
        break;
      }
      case 'p': {
        RAW_CHECK_MSG(!have_flags, kPrintfFormatsHelp);
        result += AppendPointer(&buff, buff_end, va_arg(args, uptr));
        break;
      }
      case 's': {
        RAW_CHECK_MSG(!have_flags, kPrintfFormatsHelp);
        result += AppendString(&buff, buff_end, va_arg(args, char*));
        break;
      }
      case 'c': {
        RAW_CHECK_MSG(!have_flags, kPrintfFormatsHelp);
        result += AppendChar(&buff, buff_end, va_arg(args, int));
        break;
      }
      case '%' : {
        RAW_CHECK_MSG(!have_flags, kPrintfFormatsHelp);
        result += AppendChar(&buff, buff_end, '%');
        break;
      }
      default: {
        RAW_CHECK_MSG(false, kPrintfFormatsHelp);
      }
    }
  }
  RAW_CHECK(buff <= buff_end);
  AppendChar(&buff, buff_end + 1, '\0');
  return result;
}

static void (*PrintfAndReportCallback)(const char *);
void SetPrintfAndReportCallback(void (*callback)(const char *)) {
  PrintfAndReportCallback = callback;
}

#if SANITIZER_SUPPORTS_WEAK_HOOKS
// Can be overriden in frontend.
SANITIZER_WEAK_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE
void OnPrint(const char *str);
#endif

static void CallPrintfAndReportCallback(const char *str) {
#if SANITIZER_SUPPORTS_WEAK_HOOKS
  if (&OnPrint != NULL)
    OnPrint(str);
#endif
  if (PrintfAndReportCallback)
    PrintfAndReportCallback(str);
}

static void SharedPrintfCode(bool append_pid, const char *format,
                             va_list args) {
  va_list args2;
  va_copy(args2, args);
  const int kLen = 16 * 1024;
  // |local_buffer| is small enough not to overflow the stack and/or violate
  // the stack limit enforced by TSan (-Wframe-larger-than=512). On the other
  // hand, the bigger the buffer is, the more the chance the error report will
  // fit into it.
  char local_buffer[400];
  int needed_length;
  char *buffer = local_buffer;
  int buffer_size = ARRAY_SIZE(local_buffer);
  // First try to print a message using a local buffer, and then fall back to
  // mmaped buffer.
  for (int use_mmap = 0; use_mmap < 2; use_mmap++) {
    if (use_mmap) {
      va_end(args);
      va_copy(args, args2);
      buffer = (char*)MmapOrDie(kLen, "Report");
      buffer_size = kLen;
    }
    needed_length = 0;
    if (append_pid) {
      int pid = GetPid();
      needed_length += internal_snprintf(buffer, buffer_size, "==%d==", pid);
      if (needed_length >= buffer_size) {
        // The pid doesn't fit into the current buffer.
        if (!use_mmap)
          continue;
        RAW_CHECK_MSG(needed_length < kLen, "Buffer in Report is too short!\n");
      }
    }
    needed_length += VSNPrintf(buffer + needed_length,
                               buffer_size - needed_length, format, args);
    if (needed_length >= buffer_size) {
      // The message doesn't fit into the current buffer.
      if (!use_mmap)
        continue;
      RAW_CHECK_MSG(needed_length < kLen, "Buffer in Report is too short!\n");
    }
    // If the message fit into the buffer, print it and exit.
    break;
  }
  RawWrite(buffer);
  CallPrintfAndReportCallback(buffer);
  // If we had mapped any memory, clean up.
  if (buffer != local_buffer)
    UnmapOrDie((void *)buffer, buffer_size);
  va_end(args2);
}

void Printf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  SharedPrintfCode(false, format, args);
  va_end(args);
}

// Like Printf, but prints the current PID before the output string.
void Report(const char *format, ...) {
  va_list args;
  va_start(args, format);
  SharedPrintfCode(true, format, args);
  va_end(args);
}

// Writes at most "length" symbols to "buffer" (including trailing '\0').
// Returns the number of symbols that should have been written to buffer
// (not including trailing '\0'). Thus, the string is truncated
// iff return value is not less than "length".
int internal_snprintf(char *buffer, uptr length, const char *format, ...) {
  va_list args;
  va_start(args, format);
  int needed_length = VSNPrintf(buffer, length, format, args);
  va_end(args);
  return needed_length;
}

}  // namespace __sanitizer
