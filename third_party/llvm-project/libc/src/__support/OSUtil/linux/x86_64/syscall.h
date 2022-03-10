//===---------- inline implementation of x86_64 syscalls ----------* C++ *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_X86_64_SYSCALL_H
#define LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_X86_64_SYSCALL_H

#include "src/__support/common.h"

#define SYSCALL_CLOBBER_LIST "rcx", "r11", "memory"

namespace __llvm_libc {

__attribute__((always_inline)) inline long syscall(long __number) {
  long retcode;
  LIBC_INLINE_ASM("syscall"
                  : "=a"(retcode)
                  : "a"(__number)
                  : SYSCALL_CLOBBER_LIST);
  return retcode;
}

__attribute__((always_inline)) inline long syscall(long __number, long __arg1) {
  long retcode;
  LIBC_INLINE_ASM("syscall"
                  : "=a"(retcode)
                  : "a"(__number), "D"(__arg1)
                  : SYSCALL_CLOBBER_LIST);
  return retcode;
}

__attribute__((always_inline)) inline long syscall(long __number, long __arg1,
                                                   long __arg2) {
  long retcode;
  LIBC_INLINE_ASM("syscall"
                  : "=a"(retcode)
                  : "a"(__number), "D"(__arg1), "S"(__arg2)
                  : SYSCALL_CLOBBER_LIST);
  return retcode;
}

__attribute__((always_inline)) inline long syscall(long __number, long __arg1,
                                                   long __arg2, long __arg3) {
  long retcode;
  LIBC_INLINE_ASM("syscall"
                  : "=a"(retcode)
                  : "a"(__number), "D"(__arg1), "S"(__arg2), "d"(__arg3)
                  : SYSCALL_CLOBBER_LIST);
  return retcode;
}

__attribute__((always_inline)) inline long
syscall(long __number, long __arg1, long __arg2, long __arg3, long __arg4) {
  long retcode;
  register long r10 __asm__("r10") = __arg4;
  LIBC_INLINE_ASM("syscall"
                  : "=a"(retcode)
                  : "a"(__number), "D"(__arg1), "S"(__arg2), "d"(__arg3),
                    "r"(r10)
                  : SYSCALL_CLOBBER_LIST);
  return retcode;
}

__attribute__((always_inline)) inline long syscall(long __number, long __arg1,
                                                   long __arg2, long __arg3,
                                                   long __arg4, long __arg5) {
  long retcode;
  register long r10 __asm__("r10") = __arg4;
  register long r8 __asm__("r8") = __arg5;
  LIBC_INLINE_ASM("syscall"
                  : "=a"(retcode)
                  : "a"(__number), "D"(__arg1), "S"(__arg2), "d"(__arg3),
                    "r"(r10), "r"(r8)
                  : SYSCALL_CLOBBER_LIST);
  return retcode;
}

__attribute__((always_inline)) inline long syscall(long __number, long __arg1,
                                                   long __arg2, long __arg3,
                                                   long __arg4, long __arg5,
                                                   long __arg6) {
  long retcode;
  register long r10 __asm__("r10") = __arg4;
  register long r8 __asm__("r8") = __arg5;
  register long r9 __asm__("r9") = __arg6;
  LIBC_INLINE_ASM("syscall"
                  : "=a"(retcode)
                  : "a"(__number), "D"(__arg1), "S"(__arg2), "d"(__arg3),
                    "r"(r10), "r"(r8), "r"(r9)
                  : SYSCALL_CLOBBER_LIST);
  return retcode;
}

#undef SYSCALL_CLOBBER_LIST

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_X86_64_SYSCALL_H
