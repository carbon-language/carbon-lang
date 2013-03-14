//===-- sanitizer_platform_limits_posix.cc --------------------------------===//
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

#if defined(__linux__) || defined(__APPLE__)

#include "sanitizer_internal_defs.h"
#include "sanitizer_platform_limits_posix.h"

#include <dirent.h>
#include <pthread.h>
#include <sys/utsname.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <time.h>

#if defined(__linux__)
#include <sys/vfs.h>
#include <sys/epoll.h>
#endif // __linux__

namespace __sanitizer {
  unsigned struct_utsname_sz = sizeof(struct utsname);
  unsigned struct_stat_sz = sizeof(struct stat);
  unsigned struct_stat64_sz = sizeof(struct stat64);
  unsigned struct_rusage_sz = sizeof(struct rusage);
  unsigned struct_tm_sz = sizeof(struct tm);

#if defined(__linux__)
  unsigned struct_rlimit_sz = sizeof(struct rlimit);
  unsigned struct_dirent_sz = sizeof(struct dirent);
  unsigned struct_statfs_sz = sizeof(struct statfs);
  unsigned struct_epoll_event_sz = sizeof(struct epoll_event);
#endif // __linux__

#if defined(__linux__) && !defined(__ANDROID__)
  unsigned struct_dirent64_sz = sizeof(struct dirent64);
  unsigned struct_rlimit64_sz = sizeof(struct rlimit64);
  unsigned struct_statfs64_sz = sizeof(struct statfs64);
#endif // __linux__ && !__ANDROID__

  void* __sanitizer_get_msghdr_iov_iov_base(void* msg, int idx) {
    return ((struct msghdr *)msg)->msg_iov[idx].iov_base;
  }

  uptr __sanitizer_get_msghdr_iov_iov_len(void* msg, int idx) {
    return ((struct msghdr *)msg)->msg_iov[idx].iov_len;
  }

  uptr __sanitizer_get_msghdr_iovlen(void* msg) {
    return ((struct msghdr *)msg)->msg_iovlen;
  }

  uptr __sanitizer_get_socklen_t(void* socklen_ptr) {
    return *(socklen_t*)socklen_ptr;
  }
}  // namespace __sanitizer

COMPILER_CHECK(sizeof(__sanitizer_pthread_attr_t) >= sizeof(pthread_attr_t));

#endif  // __linux__ || __APPLE__
