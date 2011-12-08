/* ===-- limits.h - stub SDK header for compiler-rt -------------------------===
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

#ifndef __SYS_MMAN_H__
#define __SYS_MMAN_H__

typedef __SIZE_TYPE__ size_t;

#define PROT_READ 0x1
#define PROT_WRITE 0x2
#define PROT_EXEC 0x4

extern int mprotect (void *__addr, size_t __len, int __prot)
  __attribute__((__nothrow__));

#endif /* __SYS_MMAN_H__ */
