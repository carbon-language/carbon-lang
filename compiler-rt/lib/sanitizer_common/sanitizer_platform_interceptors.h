//===-- sanitizer_platform_interceptors.h -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines macro telling whether sanitizer tools can/should intercept
// given library functions on a given platform.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_internal_defs.h"

#if !defined(_WIN32)
# define SANITIZER_INTERCEPT_READ 1
# define SANITIZER_INTERCEPT_PREAD 1
#else
# define SANITIZER_INTERCEPT_READ 0
# define SANITIZER_INTERCEPT_PREAD 0
#endif

#if defined(__linux__) && !defined(ANDROID)
# define SANITIZER_INTERCEPT_PREAD64 1
#else
# define SANITIZER_INTERCEPT_PREAD64 0
#endif


