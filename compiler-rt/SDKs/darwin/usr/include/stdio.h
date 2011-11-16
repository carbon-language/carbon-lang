/* ===-- stdio.h - stub SDK header for compiler-rt --------------------------===
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

#ifndef __STDIO_H__
#define __STDIO_H__

typedef struct __sFILE FILE;
typedef __SIZE_TYPE__ size_t;

/* Determine the appropriate fopen() and fwrite() functions. */
#if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__)
#  if defined(__i386)
#    define __FOPEN_NAME "_fopen$UNIX2003"
#    define __FWRITE_NAME "_fwrite$UNIX2003"
#  elif defined(__x86_64__)
#    define __FOPEN_NAME "_fopen"
#    define __FWRITE_NAME "_fwrite"
#  elif defined(__arm)
#    define __FOPEN_NAME "_fopen"
#    define __FWRITE_NAME "_fwrite"
#  else
#    error "unrecognized architecture for targetting OS X"
#  endif
#elif defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__)
#  if defined(__i386) || defined (__x86_64)
#    define __FOPEN_NAME "_fopen"
#    define __FWRITE_NAME "_fwrite"
#  elif defined(__arm)
#    define __FOPEN_NAME "_fopen"
#    define __FWRITE_NAME "_fwrite"
#  else
#    error "unrecognized architecture for targetting iOS"
#  endif
#else
#  error "unrecognized architecture for targetting Darwin"
#endif

#    define stderr __stderrp
extern FILE *__stderrp;

int fclose(FILE *);
int fflush(FILE *);
FILE *fopen(const char * restrict, const char * restrict) __asm(__FOPEN_NAME);
int fprintf(FILE * restrict, const char * restrict, ...);
size_t fwrite(const void * restrict, size_t, size_t, FILE * restrict)
  __asm(__FWRITE_NAME);

#endif /* __STDIO_H__ */
