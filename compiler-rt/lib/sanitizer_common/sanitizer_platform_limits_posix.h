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

namespace __sanitizer {
  extern unsigned struct_utsname_sz;
  extern unsigned struct_stat_sz;
  extern unsigned struct_stat64_sz;
  extern unsigned struct_rusage_sz;
  extern unsigned struct_tm_sz;

#if defined(__linux__)
  extern unsigned struct_rlimit_sz;
  extern unsigned struct_dirent_sz;
  extern unsigned struct_statfs_sz;
  extern unsigned struct_epoll_event_sz;
#endif // __linux__

#if defined(__linux__) && !defined(__ANDROID__)
  extern unsigned struct_dirent64_sz;
  extern unsigned struct_rlimit64_sz;
  extern unsigned struct_statfs64_sz;
#endif // __linux__ && !__ANDROID__

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
}  // namespace __sanitizer

#endif
