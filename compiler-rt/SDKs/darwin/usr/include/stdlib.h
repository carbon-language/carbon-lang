/* ===-- stdlib.h - stub SDK header for compiler-rt -------------------------===
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

#ifndef __STDLIB_H__
#define __STDLIB_H__

#define NULL ((void *)0)

typedef __SIZE_TYPE__ size_t;

void abort(void) __attribute__((__noreturn__));
int atexit(void (*)(void));
int atoi(const char *);
void free(void *);
char *getenv(const char *);
void *malloc(size_t);

#endif /* __STDLIB_H__ */
