//===-- sanitizer_syscall_generic.inc ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Generic implementations of internal_syscall and internal_iserror.
//
//===----------------------------------------------------------------------===//

#if SANITIZER_FREEBSD
# define SYSCALL(name) SYS_ ## name
#else
# define SYSCALL(name) __NR_ ## name
#endif

#if SANITIZER_FREEBSD && defined(__x86_64__)
# define internal_syscall __syscall
# else
# define internal_syscall syscall
#endif

bool internal_iserror(uptr retval, int *rverrno) {
  if (retval == (uptr)-1) {
    if (rverrno)
      *rverrno = errno;
    return true;
  } else {
    return false;
  }
}
