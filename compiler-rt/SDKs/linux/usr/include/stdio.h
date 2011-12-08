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

typedef __SIZE_TYPE__ size_t;

struct _IO_FILE;
typedef struct _IO_FILE FILE;

extern struct _IO_FILE *stdin;
extern struct _IO_FILE *stdout;
extern struct _IO_FILE *stderr;

extern int fclose(FILE *);
extern int fflush(FILE *);
extern FILE *fopen(const char * restrict, const char * restrict);
extern int fprintf(FILE * restrict, const char * restrict, ...);
extern size_t fwrite(const void * restrict, size_t, size_t, FILE * restrict);

#endif /* __STDIO_H__ */
