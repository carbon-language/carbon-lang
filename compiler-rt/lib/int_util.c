/* ===-- int_util.c - Implement internal utilities --------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 */

#include "int_util.h"
#include "int_lib.h"

#ifdef KERNEL_USE

extern void panic(const char *, ...) __attribute__((noreturn));
__attribute__((visibility("hidden")))
void compilerrt_abort_impl(const char *file, int line, const char *function) {
  panic("%s:%d: abort in %s", file, line, function);
}

#else

/* Get the system definition of abort() */
#include <stdlib.h>

__attribute__((visibility("hidden")))
void compilerrt_abort_impl(const char *file, int line, const char *function) {
  abort();
}

#endif
