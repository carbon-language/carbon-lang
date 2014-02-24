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

#if !defined(__linux__) && !defined(__FreeBSD__) && \
  !defined(__APPLE__) && !defined(_WIN32)
# error "This operating system is not supported"
#endif

#if defined(__linux__)
# define SANITIZER_LINUX   1
#else
# define SANITIZER_LINUX   0
#endif

#if defined(__FreeBSD__)
# define SANITIZER_FREEBSD 1
#else
# define SANITIZER_FREEBSD 0
#endif

#if defined(__APPLE__)
# define SANITIZER_MAC     1
# include <TargetConditionals.h>
# if TARGET_OS_IPHONE
#  define SANITIZER_IOS    1
# else
#  define SANITIZER_IOS    0
# endif
#else
# define SANITIZER_MAC     0
# define SANITIZER_IOS     0
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

#if __LP64__ || defined(_WIN64)
#  define SANITIZER_WORDSIZE 64
#else
#  define SANITIZER_WORDSIZE 32
#endif

#if SANITIZER_WORDSIZE == 64
# define FIRST_32_SECOND_64(a, b) (b)
#else
# define FIRST_32_SECOND_64(a, b) (a)
#endif

// By default we allow to use SizeClassAllocator64 on 64-bit platform.
// But in some cases (e.g. AArch64's 39-bit address space) SizeClassAllocator64
// does not work well and we need to fallback to SizeClassAllocator32.
// For such platforms build this code with -DSANITIZER_CAN_USE_ALLOCATOR64=0 or
// change the definition of SANITIZER_CAN_USE_ALLOCATOR64 here.
#ifndef SANITIZER_CAN_USE_ALLOCATOR64
# if defined(__aarch64__)
#  define SANITIZER_CAN_USE_ALLOCATOR64 0
# else
#  define SANITIZER_CAN_USE_ALLOCATOR64 (SANITIZER_WORDSIZE == 64)
# endif
#endif

// The range of addresses which can be returned my mmap.
// FIXME: this value should be different on different platforms,
// e.g. on AArch64 it is most likely (1ULL << 39). Larger values will still work
// but will consume more memory for TwoLevelByteMap.
#if defined(__aarch64__)
# define SANITIZER_MMAP_RANGE_SIZE FIRST_32_SECOND_64(1ULL << 32, 1ULL << 39)
#else
# define SANITIZER_MMAP_RANGE_SIZE FIRST_32_SECOND_64(1ULL << 32, 1ULL << 47)
#endif

// The AArch64 linux port uses the canonical syscall set as mandated by
// the upstream linux community for all new ports. Other ports may still
// use legacy syscalls.
#ifndef SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
# ifdef __aarch64__
# define SANITIZER_USES_CANONICAL_LINUX_SYSCALLS 1
# else
# define SANITIZER_USES_CANONICAL_LINUX_SYSCALLS 0
# endif
#endif

#endif // SANITIZER_PLATFORM_H
