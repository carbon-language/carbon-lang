//===-- sanitizer_platform.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Common platform macros.
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_PLATFORM_H
#define SANITIZER_PLATFORM_H

#if !defined(__linux__) && !defined(__APPLE__) && !defined(_WIN32)
# error "This operating system is not supported"
#endif

#if defined(__linux__)
# define SANITIZER_LINUX   1
#else
# define SANITIZER_LINUX   0
#endif

#if defined(__APPLE__)
# define SANITIZER_MAC     1
#else
# define SANITIZER_MAC     0
#endif

#if defined(_WIN32)
# define SANITIZER_WINDOWS 1
#else
# define SANITIZER_WINDOWS 0
#endif

#if defined(__ANDROID__) || defined(ANDROID)
# define SANITIZER_ANDROID 1
#else
# define SANITIZER_ANDROID 0
#endif

#define SANITIZER_POSIX (SANITIZER_LINUX || SANITIZER_MAC)

#endif // SANITIZER_PLATFORM_H
