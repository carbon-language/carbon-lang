/* ===-- string.h - stub SDK header for compiler-rt -------------------------===
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

#ifndef __STRING_H__
#define __STRING_H__

typedef __SIZE_TYPE__ size_t;

int memcmp(const void *, const void *, size_t);
void *memcpy(void *, const void *, size_t);
void *memset(void *, int, size_t);
char *strcat(char *, const char *);
char *strcpy(char *, const char *);
char *strdup(const char *);
size_t strlen(const char *);
char *strncpy(char *, const char *, size_t);

/* Determine the appropriate strerror() function. */
#if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__)
#  if defined(__i386)
#    define __STRERROR_NAME  "_strerror$UNIX2003"
#  elif defined(__x86_64__) || defined(__arm)
#    define __STRERROR_NAME  "_strerror"
#  else
#    error "unrecognized architecture for targeting OS X"
#  endif
#elif defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__)
#  if defined(__i386) || defined (__x86_64) || defined(__arm)
#    define __STRERROR_NAME  "_strerror"
#  else
#    error "unrecognized architecture for targeting iOS"
#  endif
#else
#  error "unrecognized architecture for targeting Darwin"
#endif

char *strerror(int) __asm(__STRERROR_NAME);

#endif /* __STRING_H__ */
