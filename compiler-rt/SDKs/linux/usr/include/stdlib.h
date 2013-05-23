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

void abort(void) __attribute__((__nothrow__)) __attribute__((__noreturn__));
int atexit(void (*)(void)) __attribute__((__nothrow__));
int atoi(const char *) __attribute__((__nothrow__));
void free(void *) __attribute__((__nothrow__));
char *getenv(const char *) __attribute__((__nothrow__))
  __attribute__((__nonnull__(1)));
  __attribute__((__warn_unused_result__));
void *malloc(size_t) __attribute__((__nothrow__)) __attribute((__malloc__))
     __attribute__((__warn_unused_result__));
void *realloc(void *, size_t) __attribute__((__nothrow__)) __attribute((__malloc__))
     __attribute__((__warn_unused_result__));

#endif /* __STDLIB_H__ */
