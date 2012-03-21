//===-- asan_interceptors.h -------------------------------------*- C++ -*-===//
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
// ASan-private header for asan_interceptors.cc
//===----------------------------------------------------------------------===//
#ifndef ASAN_INTERCEPTORS_H
#define ASAN_INTERCEPTORS_H

#include "asan_internal.h"
#include "interception/interception.h"

DECLARE_REAL(int, memcmp, const void *a1, const void *a2, size_t size);
DECLARE_REAL(void*, memcpy, void *to, const void *from, size_t size);
DECLARE_REAL(void*, memset, void *block, int c, size_t size);
DECLARE_REAL(char*, strchr, const char *str, int c);
DECLARE_REAL(size_t, strlen, const char *s);
DECLARE_REAL(char*, strncpy, char *to, const char *from, size_t size);
DECLARE_REAL(size_t, strnlen, const char *s, size_t maxlen);
struct sigaction;
DECLARE_REAL(int, sigaction, int signum, const struct sigaction *act,
                             struct sigaction *oldact);

namespace __asan {

// __asan::internal_X() is the implementation of X() for use in RTL.
int64_t internal_atoll(const char *nptr);
size_t internal_strlen(const char *s);
size_t internal_strnlen(const char *s, size_t maxlen);
char* internal_strchr(const char *s, int c);
void* internal_memchr(const void* s, int c, size_t n);
int internal_memcmp(const void* s1, const void* s2, size_t n);
char *internal_strstr(const char *haystack, const char *needle);
char *internal_strncat(char *dst, const char *src, size_t n);
int internal_strcmp(const char *s1, const char *s2);
char *internal_strncpy(char *dst, const char *src, size_t n);
// Works only for base=10 and doesn't set errno.
int64_t internal_simple_strtoll(const char *nptr, char **endptr, int base);

void InitializeAsanInterceptors();

#if defined(__APPLE__)
void InitializeMacInterceptors();
#endif  // __APPLE__

}  // namespace __asan

#endif  // ASAN_INTERCEPTORS_H
