/* ===-- stat.h - stub SDK header for compiler-rt ---------------------------===
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

#ifndef __SYS_STAT_H__
#define __SYS_STAT_H__

typedef unsigned short uint16_t;
typedef uint16_t mode_t;

int mkdir(const char *, mode_t);

#endif /* __SYS_STAT_H__ */
