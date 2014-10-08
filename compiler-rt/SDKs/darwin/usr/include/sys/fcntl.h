/* ===-- fcntl.h - stub SDK header for compiler-rt --------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===-----------------------------------------------------------------------===
 *
 * This is a stub SDK header file. This file is not part of the interface of
 * this library nor an official version of the appropriate SDK header. It is
 * intended only to stub the features of this header required by compiler-rt.
 *
 * ===-----------------------------------------------------------------------===
 */

#ifndef _SYS_FCNTL_H_
#define _SYS_FCNTL_H_

/* Determine the appropriate open function. */
#if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__)
#  if defined(__i386)
#    define __OPEN_NAME  "_open$UNIX2003"
#  elif defined(__x86_64__)
#    define __OPEN_NAME  "_open"
#  elif defined(__arm) || defined(__arm64)
#    define __OPEN_NAME  "_open"
#  else
#    error "unrecognized architecture for targeting OS X"
#  endif
#elif defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__)
#  if defined(__i386) || defined (__x86_64)
#    define __OPEN_NAME  "_open"
#  elif defined(__arm) || defined(__arm64)
#    define __OPEN_NAME  "_open"
#  else
#    error "unrecognized architecture for targeting iOS"
#  endif
#else
#  error "unrecognized architecture for targeting Darwin"
#endif

#define O_RDONLY   0x0000    /* open for reading only */
#define O_WRONLY   0x0001    /* open for writing only */
#define O_RDWR     0x0002    /* open for reading and writing */
#define O_ACCMODE  0x0003    /* mask for above modes */

#define O_CREAT    0x0200    /* create if nonexistent */

int open(const char *, int, ...) __asm(__OPEN_NAME);

#endif /* !_SYS_FCNTL_H_ */
