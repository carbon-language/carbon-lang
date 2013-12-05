//===-- interception_type_test.cc -------------------------------*- C++ -*-===//
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
// Compile-time tests of the internal type definitions.
//===----------------------------------------------------------------------===//

#if defined(__linux__) || defined(__APPLE__)

#include "interception.h"
#include <sys/types.h>
#include <stddef.h>
#include <stdint.h>

COMPILER_CHECK(sizeof(::SIZE_T) == sizeof(size_t));
COMPILER_CHECK(sizeof(::SSIZE_T) == sizeof(ssize_t));
COMPILER_CHECK(sizeof(::PTRDIFF_T) == sizeof(ptrdiff_t));
COMPILER_CHECK(sizeof(::INTMAX_T) == sizeof(intmax_t));

#ifndef __APPLE__
COMPILER_CHECK(sizeof(::OFF64_T) == sizeof(off64_t));
#endif

// The following are the cases when pread (and friends) is used instead of
// pread64. In those cases we need OFF_T to match off_t. We don't care about the
// rest (they depend on _FILE_OFFSET_BITS setting when building an application).
# if defined(__ANDROID__) || !defined _FILE_OFFSET_BITS || \
  _FILE_OFFSET_BITS != 64
COMPILER_CHECK(sizeof(::OFF_T) == sizeof(off_t));
# endif

#endif
