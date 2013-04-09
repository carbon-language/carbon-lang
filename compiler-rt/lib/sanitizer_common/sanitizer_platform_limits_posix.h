//===-- sanitizer_platform_limits_posix.h ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of Sanitizer common code.
//
// Sizes and layouts of platform-specific POSIX data structures.
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_PLATFORM_LIMITS_POSIX_H
#define SANITIZER_PLATFORM_LIMITS_POSIX_H

#include "sanitizer_platform.h"

namespace __sanitizer {
  extern unsigned struct_utsname_sz;
  extern unsigned struct_stat_sz;
  extern unsigned struct_stat64_sz;
  extern unsigned struct_rusage_sz;
  extern unsigned struct_tm_sz;
  extern unsigned struct_passwd_sz;
  extern unsigned struct_sigaction_sz;
  extern unsigned siginfo_t_sz;
  extern unsigned struct_itimerval_sz;
  extern unsigned pthread_t_sz;

#if !SANITIZER_ANDROID
  extern unsigned ucontext_t_sz;
#endif // !SANITIZER_ANDROID

#if SANITIZER_LINUX
  extern unsigned struct_rlimit_sz;
  extern unsigned struct_dirent_sz;
  extern unsigned struct_statfs_sz;
  extern unsigned struct_epoll_event_sz;
  extern unsigned struct_timespec_sz;
#endif // SANITIZER_LINUX

#if SANITIZER_LINUX && !SANITIZER_ANDROID
  extern unsigned struct_dirent64_sz;
  extern unsigned struct_rlimit64_sz;
  extern unsigned struct_statfs64_sz;
#endif // SANITIZER_LINUX && !SANITIZER_ANDROID

  void* __sanitizer_get_msghdr_iov_iov_base(void* msg, int idx);
  uptr __sanitizer_get_msghdr_iov_iov_len(void* msg, int idx);
  uptr __sanitizer_get_msghdr_iovlen(void* msg);
  uptr __sanitizer_get_socklen_t(void* socklen_ptr);

  // This thing depends on the platform. We are only interested in the upper
  // limit. Verified with a compiler assert in .cc.
  const int pthread_attr_t_max_sz = 128;
  union __sanitizer_pthread_attr_t {
    char size[pthread_attr_t_max_sz]; // NOLINT
    void *align;
  };

  uptr __sanitizer_get_sigaction_sa_sigaction(void *act);
  void __sanitizer_set_sigaction_sa_sigaction(void *act, uptr cb);
  bool __sanitizer_get_sigaction_sa_siginfo(void *act);

  const unsigned struct_sigaction_max_sz = 256;
  union __sanitizer_sigaction {
    char size[struct_sigaction_max_sz]; // NOLINT
  };

  extern uptr sig_ign;
  extern uptr sig_dfl;
}  // namespace __sanitizer

#endif
