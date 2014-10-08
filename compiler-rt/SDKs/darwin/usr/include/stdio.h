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

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct __sFILE FILE;
typedef __SIZE_TYPE__ size_t;

/* Determine the appropriate fdopen, fopen(), and fwrite() functions. */
#if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__)
#  if defined(__i386)
#    define __FDOPEN_NAME  "_fdopen$UNIX2003"
#    define __FOPEN_NAME "_fopen$UNIX2003"
#    define __FWRITE_NAME "_fwrite$UNIX2003"
#  elif defined(__x86_64__)
#    define __FDOPEN_NAME  "_fdopen"
#    define __FOPEN_NAME "_fopen"
#    define __FWRITE_NAME "_fwrite"
#  elif defined(__arm) || defined(__arm64)
#    define __FDOPEN_NAME  "_fdopen"
#    define __FOPEN_NAME "_fopen"
#    define __FWRITE_NAME "_fwrite"
#  else
#    error "unrecognized architecture for targeting OS X"
#  endif
#elif defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__)
#  if defined(__i386) || defined (__x86_64)
#    define __FDOPEN_NAME  "_fdopen"
#    define __FOPEN_NAME "_fopen"
#    define __FWRITE_NAME "_fwrite"
#  elif defined(__arm) || defined(__arm64)
#    define __FDOPEN_NAME  "_fdopen"
#    define __FOPEN_NAME "_fopen"
#    define __FWRITE_NAME "_fwrite"
#  else
#    error "unrecognized architecture for targeting iOS"
#  endif
#else
#  error "unrecognized architecture for targeting Darwin"
#endif

#    define stderr __stderrp
extern FILE *__stderrp;

#ifndef SEEK_SET
#define	SEEK_SET	0	/* set file offset to offset */
#endif
#ifndef SEEK_CUR
#define	SEEK_CUR	1	/* set file offset to current plus offset */
#endif
#ifndef SEEK_END
#define	SEEK_END	2	/* set file offset to EOF plus offset */
#endif

int fclose(FILE *);
int fflush(FILE *);
FILE *fopen(const char * __restrict, const char * __restrict) __asm(__FOPEN_NAME);
FILE *fdopen(int, const char *) __asm(__FDOPEN_NAME);
int fprintf(FILE * __restrict, const char * __restrict, ...);
int fputc(int, FILE *);
size_t fwrite(const void * __restrict, size_t, size_t, FILE * __restrict)
  __asm(__FWRITE_NAME);
size_t fread(void * __restrict, size_t, size_t, FILE * __restrict);
long ftell(FILE *);
int fseek(FILE *, long, int);
int snprintf(char * __restrict, size_t, const char * __restrict, ...);

#if defined(__cplusplus)
}
#endif

#endif /* __STDIO_H__ */
