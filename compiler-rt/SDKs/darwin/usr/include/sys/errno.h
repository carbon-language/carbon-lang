/* ===-- errno.h - stub SDK header for compiler-rt --------------------------===
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

#ifndef _SYS_ERRNO_H_
#define _SYS_ERRNO_H_

#if defined(__cplusplus)
extern "C" {
#endif

extern int *__error(void);
#define errno (*__error())

#if defined(__cplusplus)
}
#endif

#endif
