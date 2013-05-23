/* ===-- mman.h - stub SDK header for compiler-rt ---------------------------===
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

#define PROT_NONE     0x00
#define PROT_READ     0x01
#define PROT_WRITE    0x02
#define PROT_EXEC     0x04

#define MAP_SHARED    0x0001
#define MAP_PRIVATE   0x0002

#define MAP_FILE      0x0000
#define MAP_ANON      0x1000

#define MS_ASYNC      0x0001
#define MS_INVALIDATE 0x0002
#define MS_SYNC       0x0010

void *mmap(void *addr, size_t len, int prot, int flags, int fd,
           long long offset);
int munmap(void *addr, size_t len);
int msync(void *addr, size_t len, int flags);

#endif /* __SYS_MMAN_H__ */
