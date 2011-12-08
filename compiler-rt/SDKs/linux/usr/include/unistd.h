/* ===-- unistd.h - stub SDK header for compiler-rt -------------------------===
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

#ifndef __UNISTD_H__
#define __UNISTD_H__

enum {
  _SC_PAGESIZE = 30
};

extern long int sysconf (int __name) __attribute__ ((__nothrow__));

#endif /* __UNISTD_H__ */
