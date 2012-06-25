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

DECLARE_REAL(int, memcmp, const void *a1, const void *a2, uptr size)
DECLARE_REAL(void*, memcpy, void *to, const void *from, uptr size)
DECLARE_REAL(void*, memset, void *block, int c, uptr size)
DECLARE_REAL(char*, strchr, const char *str, int c)
DECLARE_REAL(uptr, strlen, const char *s)
DECLARE_REAL(char*, strncpy, char *to, const char *from, uptr size)
DECLARE_REAL(uptr, strnlen, const char *s, uptr maxlen)
struct sigaction;
DECLARE_REAL(int, sigaction, int signum, const struct sigaction *act,
                             struct sigaction *oldact)

namespace __asan {

void InitializeAsanInterceptors();
#if defined(__APPLE__)
void InitializeMacInterceptors();
#endif  // __APPLE__

}  // namespace __asan

#endif  // ASAN_INTERCEPTORS_H
