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

#if !SANITIZER_WINDOWS
# define SI_NOT_WINDOWS 1
# include "sanitizer_platform_limits_posix.h"
#else
# define SI_NOT_WINDOWS 0
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID
# define SI_LINUX_NOT_ANDROID 1
#else
# define SI_LINUX_NOT_ANDROID 0
#endif

#if SANITIZER_LINUX
# define SI_LINUX 1
#else
# define SI_LINUX 0
#endif

# define SANITIZER_INTERCEPT_STRCASECMP SI_NOT_WINDOWS

#if SANITIZER_MAC
# define SI_MAC 1
#else
# define SI_MAC 0
#endif

# define SANITIZER_INTERCEPT_READ   SI_NOT_WINDOWS
# define SANITIZER_INTERCEPT_PREAD  SI_NOT_WINDOWS
# define SANITIZER_INTERCEPT_WRITE  SI_NOT_WINDOWS
# define SANITIZER_INTERCEPT_PWRITE SI_NOT_WINDOWS

# define SANITIZER_INTERCEPT_PREAD64 SI_LINUX_NOT_ANDROID
# define SANITIZER_INTERCEPT_PWRITE64 SI_LINUX_NOT_ANDROID
# define SANITIZER_INTERCEPT_PRCTL   SI_LINUX

# define SANITIZER_INTERCEPT_LOCALTIME_AND_FRIENDS SI_NOT_WINDOWS

# define SANITIZER_INTERCEPT_SCANF SI_NOT_WINDOWS
# define SANITIZER_INTERCEPT_ISOC99_SCANF SI_LINUX

# define SANITIZER_INTERCEPT_FREXP 1
# define SANITIZER_INTERCEPT_FREXPF_FREXPL SI_NOT_WINDOWS

# define SANITIZER_INTERCEPT_GETPWNAM_GETPWUID SI_NOT_WINDOWS
# define SANITIZER_INTERCEPT_GETPWNAM_R_GETPWUID_R \
    SI_MAC || SI_LINUX_NOT_ANDROID
# define SANITIZER_INTERCEPT_CLOCK_GETTIME SI_LINUX
# define SANITIZER_INTERCEPT_GETITIMER SI_NOT_WINDOWS
# define SANITIZER_INTERCEPT_TIME SI_NOT_WINDOWS
# define SANITIZER_INTERCEPT_GLOB SI_LINUX_NOT_ANDROID
