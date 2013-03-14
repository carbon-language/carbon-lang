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
# define SI_NOT_WINDOWS 1
# include "sanitizer_platform_limits_posix.h"
#else
# define SI_NOT_WINDOWS 0
#endif

#if defined(__linux__) && !defined(ANDROID)
# define SI_LINUX_NOT_ANDROID 1
#else
# define SI_LINUX_NOT_ANDROID 0
#endif

#if defined(__linux__)
# define SI_LINUX 1
#else
# define SI_LINUX 0
#endif

# define SANITIZER_INTERCEPT_READ   SI_NOT_WINDOWS
# define SANITIZER_INTERCEPT_PREAD  SI_NOT_WINDOWS
# define SANITIZER_INTERCEPT_WRITE  SI_NOT_WINDOWS
# define SANITIZER_INTERCEPT_PWRITE SI_NOT_WINDOWS

# define SANITIZER_INTERCEPT_PREAD64 SI_LINUX_NOT_ANDROID
# define SANITIZER_INTERCEPT_PWRITE64 SI_LINUX_NOT_ANDROID
# define SANITIZER_INTERCEPT_PRCTL   SI_LINUX_NOT_ANDROID

# define SANITIZER_INTERCEPT_LOCALTIME_AND_FRIENDS SI_NOT_WINDOWS

# define SANITIZER_INTERCEPT_SCANF SI_NOT_WINDOWS
# define SANITIZER_INTERCEPT_ISOC99_SCANF SI_LINUX

# define SANITIZER_INTERCEPT_FREXP 1
