//===-- msan_platform_limits.cc -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
// Sizes and layouts of platform-specific POSIX data structures.
//===----------------------------------------------------------------------===//

#ifdef __linux__

#include "msan.h"
#include "msan_platform_limits_posix.h"

#include <sys/utsname.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/vfs.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <dirent.h>

namespace __msan {
  unsigned struct_utsname_sz = sizeof(struct utsname);
  unsigned struct_stat_sz = sizeof(struct stat);
  unsigned struct_stat64_sz = sizeof(struct stat64);
  unsigned struct_rlimit_sz = sizeof(struct rlimit);
  unsigned struct_rlimit64_sz = sizeof(struct rlimit64);
  unsigned struct_dirent_sz = sizeof(struct dirent);
  unsigned struct_statfs_sz = sizeof(struct statfs);
  unsigned struct_statfs64_sz = sizeof(struct statfs64);
  unsigned struct_epoll_event_sz = sizeof(struct epoll_event);

  void* __msan_get_msghdr_iov_iov_base(void* msg, int idx) {
    return ((struct msghdr *)msg)->msg_iov[idx].iov_base;
  }

  uptr __msan_get_msghdr_iov_iov_len(void* msg, int idx) {
    return ((struct msghdr *)msg)->msg_iov[idx].iov_len;
  }

  uptr __msan_get_msghdr_iovlen(void* msg) {
    return ((struct msghdr *)msg)->msg_iovlen;
  }
}

#endif  // __linux__
