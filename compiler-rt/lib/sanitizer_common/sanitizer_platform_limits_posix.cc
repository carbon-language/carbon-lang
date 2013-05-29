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


#include "sanitizer_platform.h"
#if SANITIZER_LINUX || SANITIZER_MAC

#include "sanitizer_internal_defs.h"
#include "sanitizer_platform_limits_posix.h"

#include <arpa/inet.h>
#include <dirent.h>
#include <grp.h>
#include <pthread.h>
#include <pwd.h>
#include <signal.h>
#include <stddef.h>
#include <sys/utsname.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <netdb.h>
#include <time.h>

#if !SANITIZER_ANDROID
#include <sys/ucontext.h>
#endif // !SANITIZER_ANDROID

#if SANITIZER_LINUX
#include <link.h>
#include <sys/vfs.h>
#include <sys/epoll.h>
#endif // SANITIZER_LINUX

namespace __sanitizer {
  unsigned struct_utsname_sz = sizeof(struct utsname);
  unsigned struct_stat_sz = sizeof(struct stat);
  unsigned struct_stat64_sz = sizeof(struct stat64);
  unsigned struct_rusage_sz = sizeof(struct rusage);
  unsigned struct_tm_sz = sizeof(struct tm);
  unsigned struct_passwd_sz = sizeof(struct passwd);
  unsigned struct_group_sz = sizeof(struct group);
  unsigned siginfo_t_sz = sizeof(siginfo_t);
  unsigned struct_sigaction_sz = sizeof(struct sigaction);
  unsigned struct_itimerval_sz = sizeof(struct itimerval);
  unsigned pthread_t_sz = sizeof(pthread_t);
  unsigned struct_sockaddr_sz = sizeof(struct sockaddr);

#if !SANITIZER_ANDROID
  unsigned ucontext_t_sz = sizeof(ucontext_t);
#endif // !SANITIZER_ANDROID

#if SANITIZER_LINUX
  unsigned struct_rlimit_sz = sizeof(struct rlimit);
  unsigned struct_dirent_sz = sizeof(struct dirent);
  unsigned struct_statfs_sz = sizeof(struct statfs);
  unsigned struct_epoll_event_sz = sizeof(struct epoll_event);
  unsigned struct_timespec_sz = sizeof(struct timespec);
#endif // SANITIZER_LINUX

#if SANITIZER_LINUX && !SANITIZER_ANDROID
  unsigned struct_dirent64_sz = sizeof(struct dirent64);
  unsigned struct_rlimit64_sz = sizeof(struct rlimit64);
  unsigned struct_statfs64_sz = sizeof(struct statfs64);
#endif // SANITIZER_LINUX && !SANITIZER_ANDROID

  uptr sig_ign = (uptr)SIG_IGN;
  uptr sig_dfl = (uptr)SIG_DFL;

  uptr __sanitizer_get_sigaction_sa_sigaction(void *act) {
    struct sigaction *a = (struct sigaction *)act;
    // Check that sa_sigaction and sa_handler are the same.
    CHECK((void *)&(a->sa_sigaction) == (void *)&(a->sa_handler));
    return (uptr) a->sa_sigaction;
  }
  void __sanitizer_set_sigaction_sa_sigaction(void *act, uptr cb) {
    struct sigaction *a = (struct sigaction *)act;
    a->sa_sigaction = (void (*)(int, siginfo_t *, void *))cb;
  }
  bool __sanitizer_get_sigaction_sa_siginfo(void *act) {
    struct sigaction *a = (struct sigaction *)act;
    return a->sa_flags & SA_SIGINFO;
  }

  uptr __sanitizer_in_addr_sz(int af) {
    if (af == AF_INET)
      return sizeof(struct in_addr);
    else if (af == AF_INET6)
      return sizeof(struct in6_addr);
    else
      return 0;
  }
}  // namespace __sanitizer

COMPILER_CHECK(sizeof(__sanitizer_pthread_attr_t) >= sizeof(pthread_attr_t));
COMPILER_CHECK(sizeof(__sanitizer::struct_sigaction_max_sz) >=
                   sizeof(__sanitizer::struct_sigaction_sz));
#if SANITIZER_LINUX
COMPILER_CHECK(offsetof(struct __sanitizer_dl_phdr_info, dlpi_addr) ==
               offsetof(struct dl_phdr_info, dlpi_addr));
COMPILER_CHECK(offsetof(struct __sanitizer_dl_phdr_info, dlpi_name) ==
               offsetof(struct dl_phdr_info, dlpi_name));
COMPILER_CHECK(offsetof(struct __sanitizer_dl_phdr_info, dlpi_phdr) ==
               offsetof(struct dl_phdr_info, dlpi_phdr));
COMPILER_CHECK(offsetof(struct __sanitizer_dl_phdr_info, dlpi_phnum) ==
               offsetof(struct dl_phdr_info, dlpi_phnum));
#endif

COMPILER_CHECK(sizeof(socklen_t) == sizeof(unsigned));

COMPILER_CHECK(sizeof(struct __sanitizer_addrinfo) == sizeof(struct addrinfo));
COMPILER_CHECK(offsetof(struct __sanitizer_addrinfo, ai_addr) ==
               offsetof(struct addrinfo, ai_addr));
COMPILER_CHECK(offsetof(struct __sanitizer_addrinfo, ai_canonname) ==
               offsetof(struct addrinfo, ai_canonname));
COMPILER_CHECK(offsetof(struct __sanitizer_addrinfo, ai_next) ==
               offsetof(struct addrinfo, ai_next));

COMPILER_CHECK(sizeof(struct __sanitizer_hostent) == sizeof(struct hostent));
COMPILER_CHECK(offsetof(struct __sanitizer_hostent, h_name) ==
               offsetof(struct hostent, h_name));
COMPILER_CHECK(offsetof(struct __sanitizer_hostent, h_aliases) ==
               offsetof(struct hostent, h_aliases));
COMPILER_CHECK(offsetof(struct __sanitizer_hostent, h_addr_list) ==
               offsetof(struct hostent, h_addr_list));

COMPILER_CHECK(sizeof(struct __sanitizer_iovec) == sizeof(struct iovec));
COMPILER_CHECK(offsetof(struct __sanitizer_iovec, iov_base) ==
               offsetof(struct iovec, iov_base));
COMPILER_CHECK(offsetof(struct __sanitizer_iovec, iov_len) ==
               offsetof(struct iovec, iov_len));

COMPILER_CHECK(sizeof(struct __sanitizer_msghdr) == sizeof(struct msghdr));
COMPILER_CHECK(offsetof(struct __sanitizer_msghdr, msg_name) ==
               offsetof(struct msghdr, msg_name));
COMPILER_CHECK(offsetof(struct __sanitizer_msghdr, msg_namelen) ==
               offsetof(struct msghdr, msg_namelen));
COMPILER_CHECK(offsetof(struct __sanitizer_msghdr, msg_iov) ==
               offsetof(struct msghdr, msg_iov));
COMPILER_CHECK(offsetof(struct __sanitizer_msghdr, msg_iovlen) ==
               offsetof(struct msghdr, msg_iovlen));
COMPILER_CHECK(offsetof(struct __sanitizer_msghdr, msg_control) ==
               offsetof(struct msghdr, msg_control));
COMPILER_CHECK(offsetof(struct __sanitizer_msghdr, msg_controllen) ==
               offsetof(struct msghdr, msg_controllen));
COMPILER_CHECK(offsetof(struct __sanitizer_msghdr, msg_flags) ==
               offsetof(struct msghdr, msg_flags));

COMPILER_CHECK(sizeof(struct __sanitizer_cmsghdr) == sizeof(struct cmsghdr));
COMPILER_CHECK(offsetof(struct __sanitizer_cmsghdr, cmsg_len) ==
               offsetof(struct cmsghdr, cmsg_len));
COMPILER_CHECK(offsetof(struct __sanitizer_cmsghdr, cmsg_level) ==
               offsetof(struct cmsghdr, cmsg_level));
COMPILER_CHECK(offsetof(struct __sanitizer_cmsghdr, cmsg_type) ==
               offsetof(struct cmsghdr, cmsg_type));

#endif  // SANITIZER_LINUX || SANITIZER_MAC
