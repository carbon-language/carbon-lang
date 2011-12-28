//===-- asan_interceptors.cc ------------------------------------*- C++ -*-===//
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
// Intercept various libc functions to catch buggy memory accesses there.
//===----------------------------------------------------------------------===//
#include "asan_interceptors.h"

#include "asan_allocator.h"
#include "asan_interface.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_stack.h"
#include "asan_stats.h"

#include <ctype.h>
#include <dlfcn.h>
#include <string.h>
#include <strings.h>

namespace __asan {

index_f       real_index;
memcmp_f      real_memcmp;
memcpy_f      real_memcpy;
memmove_f     real_memmove;
memset_f      real_memset;
strcasecmp_f  real_strcasecmp;
strchr_f      real_strchr;
strcmp_f      real_strcmp;
strcpy_f      real_strcpy;
strdup_f      real_strdup;
strlen_f      real_strlen;
strncasecmp_f real_strncasecmp;
strncmp_f     real_strncmp;
strncpy_f     real_strncpy;
strnlen_f     real_strnlen;

// Instruments read/write access to a single byte in memory.
// On error calls __asan_report_error, which aborts the program.
__attribute__((noinline))
static void AccessAddress(uintptr_t address, bool isWrite) {
  if (__asan_address_is_poisoned((void*)address)) {
    GET_BP_PC_SP;
    __asan_report_error(pc, bp, sp, address, isWrite, /* access_size */ 1);
  }
}

// We implement ACCESS_MEMORY_RANGE, ASAN_READ_RANGE,
// and ASAN_WRITE_RANGE as macro instead of function so
// that no extra frames are created, and stack trace contains
// relevant information only.

// Instruments read/write access to a memory range.
// More complex implementation is possible, for now just
// checking the first and the last byte of a range.
#define ACCESS_MEMORY_RANGE(offset, size, isWrite) do { \
  if (size > 0) { \
    uintptr_t ptr = (uintptr_t)(offset); \
    AccessAddress(ptr, isWrite); \
    AccessAddress(ptr + (size) - 1, isWrite); \
  } \
} while (0)

#define ASAN_READ_RANGE(offset, size) do { \
  ACCESS_MEMORY_RANGE(offset, size, false); \
} while (0)

#define ASAN_WRITE_RANGE(offset, size) do { \
  ACCESS_MEMORY_RANGE(offset, size, true); \
} while (0)

// Behavior of functions like "memcpy" or "strcpy" is undefined
// if memory intervals overlap. We report error in this case.
// Macro is used to avoid creation of new frames.
static inline bool RangesOverlap(const char *offset1, const char *offset2,
                                 size_t length) {
  return !((offset1 + length <= offset2) || (offset2 + length <= offset1));
}
#define CHECK_RANGES_OVERLAP(_offset1, _offset2, length) do { \
  const char *offset1 = (const char*)_offset1; \
  const char *offset2 = (const char*)_offset2; \
  if (RangesOverlap((const char*)offset1, (const char*)offset2, \
                    length)) { \
    Report("ERROR: AddressSanitizer strcpy-param-overlap: " \
           "memory ranges [%p,%p) and [%p, %p) overlap\n", \
           offset1, offset1 + length, offset2, offset2 + length); \
    PRINT_CURRENT_STACK(); \
    ShowStatsAndAbort(); \
  } \
} while (0)

#define ENSURE_ASAN_INITED() do { \
  CHECK(!asan_init_is_running); \
  if (!asan_inited) { \
    __asan_init(); \
  } \
} while (0)

size_t internal_strlen(const char *s) {
  size_t i = 0;
  while (s[i]) i++;
  return i;
}

size_t internal_strnlen(const char *s, size_t maxlen) {
  if (real_strnlen != NULL) {
    return real_strnlen(s, maxlen);
  }
  size_t i = 0;
  while (i < maxlen && s[i]) i++;
  return i;
}

void InitializeAsanInterceptors() {
#ifndef __APPLE__
  INTERCEPT_FUNCTION(index);
#else
  OVERRIDE_FUNCTION(index, WRAP(strchr));
#endif
  INTERCEPT_FUNCTION(memcmp);
  INTERCEPT_FUNCTION(memcpy);
  INTERCEPT_FUNCTION(memmove);
  INTERCEPT_FUNCTION(memset);
  INTERCEPT_FUNCTION(strcasecmp);
  INTERCEPT_FUNCTION(strchr);
  INTERCEPT_FUNCTION(strcmp);
  INTERCEPT_FUNCTION(strcpy);  // NOLINT
  INTERCEPT_FUNCTION(strdup);
  INTERCEPT_FUNCTION(strlen);
  INTERCEPT_FUNCTION(strncasecmp);
  INTERCEPT_FUNCTION(strncmp);
  INTERCEPT_FUNCTION(strncpy);
#ifndef __APPLE__
  INTERCEPT_FUNCTION(strnlen);
#endif
  if (FLAG_v > 0) {
    Printf("AddressSanitizer: libc interceptors initialized\n");
  }
}

}  // namespace __asan

// ---------------------- Wrappers ---------------- {{{1
using namespace __asan;  // NOLINT

static inline int CharCmp(unsigned char c1, unsigned char c2) {
  return (c1 == c2) ? 0 : (c1 < c2) ? -1 : 1;
}

static inline int CharCaseCmp(unsigned char c1, unsigned char c2) {
  int c1_low = tolower(c1);
  int c2_low = tolower(c2);
  return c1_low - c2_low;
}

int WRAP(memcmp)(const void *a1, const void *a2, size_t size) {
  ENSURE_ASAN_INITED();
  unsigned char c1 = 0, c2 = 0;
  const unsigned char *s1 = (const unsigned char*)a1;
  const unsigned char *s2 = (const unsigned char*)a2;
  size_t i;
  for (i = 0; i < size; i++) {
    c1 = s1[i];
    c2 = s2[i];
    if (c1 != c2) break;
  }
  ASAN_READ_RANGE(s1, Min(i + 1, size));
  ASAN_READ_RANGE(s2, Min(i + 1, size));
  return CharCmp(c1, c2);
}

void *WRAP(memcpy)(void *to, const void *from, size_t size) {
  // memcpy is called during __asan_init() from the internals
  // of printf(...).
  if (asan_init_is_running) {
    return real_memcpy(to, from, size);
  }
  ENSURE_ASAN_INITED();
  if (FLAG_replace_intrin) {
    CHECK_RANGES_OVERLAP(to, from, size);
    ASAN_WRITE_RANGE(from, size);
    ASAN_READ_RANGE(to, size);
  }
  return real_memcpy(to, from, size);
}

void *WRAP(memmove)(void *to, const void *from, size_t size) {
  ENSURE_ASAN_INITED();
  if (FLAG_replace_intrin) {
    ASAN_WRITE_RANGE(from, size);
    ASAN_READ_RANGE(to, size);
  }
  return real_memmove(to, from, size);
}

void *WRAP(memset)(void *block, int c, size_t size) {
  // memset is called inside INTERCEPT_FUNCTION on Mac.
  if (asan_init_is_running) {
    return real_memset(block, c, size);
  }
  ENSURE_ASAN_INITED();
  if (FLAG_replace_intrin) {
    ASAN_WRITE_RANGE(block, size);
  }
  return real_memset(block, c, size);
}

// Note that on Linux index and strchr are definined differently depending on
// the compiler (gcc vs clang).
// see __CORRECT_ISO_CPP_STRING_H_PROTO in /usr/include/string.h

#ifndef __APPLE__
char *WRAP(index)(const char *str, int c)
  __attribute__((alias(WRAPPER_NAME(strchr))));
#endif

char *WRAP(strchr)(const char *str, int c) {
  ENSURE_ASAN_INITED();
  char *result = real_strchr(str, c);
  if (FLAG_replace_str) {
    size_t bytes_read = (result ? result - str : real_strlen(str)) + 1;
    ASAN_READ_RANGE(str, bytes_read);
  }
  return result;
}

int WRAP(strcasecmp)(const char *s1, const char *s2) {
  ENSURE_ASAN_INITED();
  unsigned char c1, c2;
  size_t i;
  for (i = 0; ; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (CharCaseCmp(c1, c2) != 0 || c1 == '\0') break;
  }
  ASAN_READ_RANGE(s1, i + 1);
  ASAN_READ_RANGE(s2, i + 1);
  return CharCaseCmp(c1, c2);
}

int WRAP(strcmp)(const char *s1, const char *s2) {
  // strcmp is called from malloc_default_purgeable_zone()
  // in __asan::ReplaceSystemAlloc() on Mac.
  if (asan_init_is_running) {
    return real_strcmp(s1, s2);
  }
  unsigned char c1, c2;
  size_t i;
  for (i = 0; ; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (c1 != c2 || c1 == '\0') break;
  }
  ASAN_READ_RANGE(s1, i + 1);
  ASAN_READ_RANGE(s2, i + 1);
  return CharCmp(c1, c2);
}

char *WRAP(strcpy)(char *to, const char *from) {  // NOLINT
  // strcpy is called from malloc_default_purgeable_zone()
  // in __asan::ReplaceSystemAlloc() on Mac.
  if (asan_init_is_running) {
    return real_strcpy(to, from);
  }
  ENSURE_ASAN_INITED();
  if (FLAG_replace_str) {
    size_t from_size = real_strlen(from) + 1;
    CHECK_RANGES_OVERLAP(to, from, from_size);
    ASAN_READ_RANGE(from, from_size);
    ASAN_WRITE_RANGE(to, from_size);
  }
  return real_strcpy(to, from);
}

char *WRAP(strdup)(const char *s) {
  ENSURE_ASAN_INITED();
  if (FLAG_replace_str) {
    size_t length = real_strlen(s);
    ASAN_READ_RANGE(s, length + 1);
  }
  return real_strdup(s);
}

size_t WRAP(strlen)(const char *s) {
  // strlen is called from malloc_default_purgeable_zone()
  // in __asan::ReplaceSystemAlloc() on Mac.
  if (asan_init_is_running) {
    return real_strlen(s);
  }
  ENSURE_ASAN_INITED();
  size_t length = real_strlen(s);
  if (FLAG_replace_str) {
    ASAN_READ_RANGE(s, length + 1);
  }
  return length;
}

int WRAP(strncasecmp)(const char *s1, const char *s2, size_t size) {
  ENSURE_ASAN_INITED();
  unsigned char c1 = 0, c2 = 0;
  size_t i;
  for (i = 0; i < size; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (CharCaseCmp(c1, c2) != 0 || c1 == '\0') break;
  }
  ASAN_READ_RANGE(s1, Min(i + 1, size));
  ASAN_READ_RANGE(s2, Min(i + 1, size));
  return CharCaseCmp(c1, c2);
}

int WRAP(strncmp)(const char *s1, const char *s2, size_t size) {
  // strncmp is called from malloc_default_purgeable_zone()
  // in __asan::ReplaceSystemAlloc() on Mac.
  if (asan_init_is_running) {
    return real_strncmp(s1, s2, size);
  }
  unsigned char c1 = 0, c2 = 0;
  size_t i;
  for (i = 0; i < size; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (c1 != c2 || c1 == '\0') break;
  }
  ASAN_READ_RANGE(s1, Min(i + 1, size));
  ASAN_READ_RANGE(s2, Min(i + 1, size));
  return CharCmp(c1, c2);
}

char *WRAP(strncpy)(char *to, const char *from, size_t size) {
  ENSURE_ASAN_INITED();
  if (FLAG_replace_str) {
    size_t from_size = Min(size, internal_strnlen(from, size) + 1);
    CHECK_RANGES_OVERLAP(to, from, from_size);
    ASAN_READ_RANGE(from, from_size);
    ASAN_WRITE_RANGE(to, size);
  }
  return real_strncpy(to, from, size);
}

#ifndef __APPLE__
size_t WRAP(strnlen)(const char *s, size_t maxlen) {
  ENSURE_ASAN_INITED();
  size_t length = real_strnlen(s, maxlen);
  if (FLAG_replace_str) {
    ASAN_READ_RANGE(s, Min(length + 1, maxlen));
  }
  return length;
}
#endif
