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
char *strcat(char *, const char *);
char *strcpy(char *, const char *);
char *strdup(const char *);
size_t strlen(const char *);
char *strncpy(char *, const char *, size_t);

#endif /* __STRING_H__ */
