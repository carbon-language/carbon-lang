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
  extern unsigned struct_rlimit64_sz;
  extern unsigned struct_dirent_sz;
  extern unsigned struct_statfs_sz;
  extern unsigned struct_statfs64_sz;
  extern unsigned struct_epoll_event_sz;
#endif // __linux__

  void* __sanitizer_get_msghdr_iov_iov_base(void* msg, int idx);
  uptr __sanitizer_get_msghdr_iov_iov_len(void* msg, int idx);
  uptr __sanitizer_get_msghdr_iovlen(void* msg);
  uptr __sanitizer_get_socklen_t(void* socklen_ptr);
}  // namespace __sanitizer

#endif
