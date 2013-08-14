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
#include <limits.h>
#include <net/if.h>
#include <net/if_arp.h>
#include <net/route.h>
#include <netdb.h>
#include <poll.h>
#include <pthread.h>
#include <pwd.h>
#include <signal.h>
#include <stddef.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <termios.h>
#include <time.h>
#include <wchar.h>

#if SANITIZER_LINUX
#include <sys/ptrace.h>
#include <sys/sysinfo.h>
#endif

#if !SANITIZER_ANDROID
#include <sys/ucontext.h>
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID
#include <glob.h>
#include <sys/mtio.h>
#include <sys/kd.h>
#include <sys/user.h>
#endif // SANITIZER_LINUX && !SANITIZER_ANDROID

#if SANITIZER_LINUX
#include <link.h>
#include <sys/vfs.h>
#include <sys/epoll.h>
#endif // SANITIZER_LINUX

#if SANITIZER_MAC
#include <netinet/ip_mroute.h>
#include <sys/filio.h>
#include <sys/sockio.h>
#endif

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
  unsigned pid_t_sz = sizeof(pid_t);
  unsigned timeval_sz = sizeof(timeval);
  unsigned uid_t_sz = sizeof(uid_t);
  unsigned mbstate_t_sz = sizeof(mbstate_t);
  unsigned sigset_t_sz = sizeof(sigset_t);

#if !SANITIZER_ANDROID
  unsigned ucontext_t_sz = sizeof(ucontext_t);
#endif // !SANITIZER_ANDROID

#if SANITIZER_LINUX
  unsigned struct_rlimit_sz = sizeof(struct rlimit);
  unsigned struct_statfs_sz = sizeof(struct statfs);
  unsigned struct_epoll_event_sz = sizeof(struct epoll_event);
  unsigned struct_sysinfo_sz = sizeof(struct sysinfo);
  unsigned struct_timespec_sz = sizeof(struct timespec);
#endif // SANITIZER_LINUX

#if SANITIZER_LINUX && !SANITIZER_ANDROID
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

  int af_inet = (int)AF_INET;
  int af_inet6 = (int)AF_INET6;

  uptr __sanitizer_in_addr_sz(int af) {
    if (af == AF_INET)
      return sizeof(struct in_addr);
    else if (af == AF_INET6)
      return sizeof(struct in6_addr);
    else
      return 0;
  }

#if SANITIZER_LINUX && !SANITIZER_ANDROID
  int glob_nomatch = GLOB_NOMATCH;
  int glob_altdirfunc = GLOB_ALTDIRFUNC;
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID && \
      (defined(__i386) || defined (__x86_64))  // NOLINT
  unsigned struct_user_regs_struct_sz = sizeof(struct user_regs_struct);
  unsigned struct_user_fpregs_struct_sz = sizeof(struct user_fpregs_struct);
#if __WORDSIZE == 64
  unsigned struct_user_fpxregs_struct_sz = 0;
#else
  unsigned struct_user_fpxregs_struct_sz = sizeof(struct user_fpxregs_struct);
#endif

  int ptrace_getregs = PTRACE_GETREGS;
  int ptrace_setregs = PTRACE_SETREGS;
  int ptrace_getfpregs = PTRACE_GETFPREGS;
  int ptrace_setfpregs = PTRACE_SETFPREGS;
  int ptrace_getfpxregs = PTRACE_GETFPXREGS;
  int ptrace_setfpxregs = PTRACE_SETFPXREGS;
  int ptrace_getsiginfo = PTRACE_GETSIGINFO;
  int ptrace_setsiginfo = PTRACE_SETSIGINFO;
#if defined(PTRACE_GETREGSET) && defined(PTRACE_SETREGSET)
  int ptrace_getregset = PTRACE_GETREGSET;
  int ptrace_setregset = PTRACE_SETREGSET;
#else
  int ptrace_getregset = -1;
  int ptrace_setregset = -1;
#endif
#endif

  unsigned path_max = PATH_MAX;

  // ioctl arguments
  unsigned struct_arpreq_sz = sizeof(struct arpreq);
  unsigned struct_ifreq_sz = sizeof(struct ifreq);
  unsigned struct_termios_sz = sizeof(struct termios);
  unsigned struct_winsize_sz = sizeof(struct winsize);

#if SANITIZER_MAC
  unsigned struct_sioc_sg_req_sz = sizeof(struct sioc_sg_req);
  unsigned struct_sioc_vif_req_sz = sizeof(struct sioc_vif_req);
#endif

  unsigned IOCTL_NOT_PRESENT = 0;

  unsigned IOCTL_FIOASYNC = FIOASYNC;
  unsigned IOCTL_FIOCLEX = FIOCLEX;
  unsigned IOCTL_FIOGETOWN = FIOGETOWN;
  unsigned IOCTL_FIONBIO = FIONBIO;
  unsigned IOCTL_FIONCLEX = FIONCLEX;
  unsigned IOCTL_FIOSETOWN = FIOSETOWN;
  unsigned IOCTL_SIOCADDMULTI = SIOCADDMULTI;
  unsigned IOCTL_SIOCATMARK = SIOCATMARK;
  unsigned IOCTL_SIOCDELMULTI = SIOCDELMULTI;
  unsigned IOCTL_SIOCGIFADDR = SIOCGIFADDR;
  unsigned IOCTL_SIOCGIFBRDADDR = SIOCGIFBRDADDR;
  unsigned IOCTL_SIOCGIFCONF = SIOCGIFCONF;
  unsigned IOCTL_SIOCGIFDSTADDR = SIOCGIFDSTADDR;
  unsigned IOCTL_SIOCGIFFLAGS = SIOCGIFFLAGS;
  unsigned IOCTL_SIOCGIFMETRIC = SIOCGIFMETRIC;
  unsigned IOCTL_SIOCGIFMTU = SIOCGIFMTU;
  unsigned IOCTL_SIOCGIFNETMASK = SIOCGIFNETMASK;
  unsigned IOCTL_SIOCGPGRP = SIOCGPGRP;
  unsigned IOCTL_SIOCSIFADDR = SIOCSIFADDR;
  unsigned IOCTL_SIOCSIFBRDADDR = SIOCSIFBRDADDR;
  unsigned IOCTL_SIOCSIFDSTADDR = SIOCSIFDSTADDR;
  unsigned IOCTL_SIOCSIFFLAGS = SIOCSIFFLAGS;
  unsigned IOCTL_SIOCSIFMETRIC = SIOCSIFMETRIC;
  unsigned IOCTL_SIOCSIFMTU = SIOCSIFMTU;
  unsigned IOCTL_SIOCSIFNETMASK = SIOCSIFNETMASK;
  unsigned IOCTL_SIOCSPGRP = SIOCSPGRP;
  unsigned IOCTL_TIOCCONS = TIOCCONS;
  unsigned IOCTL_TIOCEXCL = TIOCEXCL;
  unsigned IOCTL_TIOCGETD = TIOCGETD;
  unsigned IOCTL_TIOCGPGRP = TIOCGPGRP;
  unsigned IOCTL_TIOCGWINSZ = TIOCGWINSZ;
  unsigned IOCTL_TIOCMBIC = TIOCMBIC;
  unsigned IOCTL_TIOCMBIS = TIOCMBIS;
  unsigned IOCTL_TIOCMGET = TIOCMGET;
  unsigned IOCTL_TIOCMSET = TIOCMSET;
  unsigned IOCTL_TIOCNOTTY = TIOCNOTTY;
  unsigned IOCTL_TIOCNXCL = TIOCNXCL;
  unsigned IOCTL_TIOCOUTQ = TIOCOUTQ;
  unsigned IOCTL_TIOCPKT = TIOCPKT;
  unsigned IOCTL_TIOCSCTTY = TIOCSCTTY;
  unsigned IOCTL_TIOCSETD = TIOCSETD;
  unsigned IOCTL_TIOCSPGRP = TIOCSPGRP;
  unsigned IOCTL_TIOCSTI = TIOCSTI;
  unsigned IOCTL_TIOCSWINSZ = TIOCSWINSZ;
#if SANITIZER_MAC
  unsigned IOCTL_SIOCGETSGCNT = SIOCGETSGCNT;
  unsigned IOCTL_SIOCGETVIFCNT = SIOCGETVIFCNT;
#endif


}  // namespace __sanitizer

#define CHECK_TYPE_SIZE(TYPE) \
  COMPILER_CHECK(sizeof(__sanitizer_##TYPE) == sizeof(TYPE))

#define CHECK_SIZE_AND_OFFSET(CLASS, MEMBER)                       \
  COMPILER_CHECK(sizeof(((__sanitizer_##CLASS *) NULL)->MEMBER) == \
                 sizeof(((CLASS *) NULL)->MEMBER));                \
  COMPILER_CHECK(offsetof(__sanitizer_##CLASS, MEMBER) ==          \
                 offsetof(CLASS, MEMBER))

COMPILER_CHECK(sizeof(__sanitizer_pthread_attr_t) >= sizeof(pthread_attr_t));
COMPILER_CHECK(sizeof(__sanitizer::struct_sigaction_max_sz) >=
                   sizeof(__sanitizer::struct_sigaction_sz));

COMPILER_CHECK(sizeof(socklen_t) == sizeof(unsigned));
CHECK_TYPE_SIZE(pthread_key_t);

#if SANITIZER_LINUX
// There are more undocumented fields in dl_phdr_info that we are not interested
// in.
COMPILER_CHECK(sizeof(__sanitizer_dl_phdr_info) <= sizeof(dl_phdr_info));
CHECK_SIZE_AND_OFFSET(dl_phdr_info, dlpi_addr);
CHECK_SIZE_AND_OFFSET(dl_phdr_info, dlpi_name);
CHECK_SIZE_AND_OFFSET(dl_phdr_info, dlpi_phdr);
CHECK_SIZE_AND_OFFSET(dl_phdr_info, dlpi_phnum);

COMPILER_CHECK(IOC_SIZE(0x12345678) == _IOC_SIZE(0x12345678));
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID
CHECK_TYPE_SIZE(glob_t);
CHECK_SIZE_AND_OFFSET(glob_t, gl_pathc);
CHECK_SIZE_AND_OFFSET(glob_t, gl_pathv);
CHECK_SIZE_AND_OFFSET(glob_t, gl_offs);
CHECK_SIZE_AND_OFFSET(glob_t, gl_flags);
CHECK_SIZE_AND_OFFSET(glob_t, gl_closedir);
CHECK_SIZE_AND_OFFSET(glob_t, gl_readdir);
CHECK_SIZE_AND_OFFSET(glob_t, gl_opendir);
CHECK_SIZE_AND_OFFSET(glob_t, gl_lstat);
CHECK_SIZE_AND_OFFSET(glob_t, gl_stat);
#endif

CHECK_TYPE_SIZE(addrinfo);
CHECK_SIZE_AND_OFFSET(addrinfo, ai_flags);
CHECK_SIZE_AND_OFFSET(addrinfo, ai_family);
CHECK_SIZE_AND_OFFSET(addrinfo, ai_socktype);
CHECK_SIZE_AND_OFFSET(addrinfo, ai_protocol);
CHECK_SIZE_AND_OFFSET(addrinfo, ai_protocol);
CHECK_SIZE_AND_OFFSET(addrinfo, ai_addrlen);
CHECK_SIZE_AND_OFFSET(addrinfo, ai_canonname);
CHECK_SIZE_AND_OFFSET(addrinfo, ai_addr);

CHECK_TYPE_SIZE(hostent);
CHECK_SIZE_AND_OFFSET(hostent, h_name);
CHECK_SIZE_AND_OFFSET(hostent, h_aliases);
CHECK_SIZE_AND_OFFSET(hostent, h_addrtype);
CHECK_SIZE_AND_OFFSET(hostent, h_length);
CHECK_SIZE_AND_OFFSET(hostent, h_addr_list);

CHECK_TYPE_SIZE(iovec);
CHECK_SIZE_AND_OFFSET(iovec, iov_base);
CHECK_SIZE_AND_OFFSET(iovec, iov_len);

CHECK_TYPE_SIZE(msghdr);
CHECK_SIZE_AND_OFFSET(msghdr, msg_name);
CHECK_SIZE_AND_OFFSET(msghdr, msg_namelen);
CHECK_SIZE_AND_OFFSET(msghdr, msg_iov);
CHECK_SIZE_AND_OFFSET(msghdr, msg_iovlen);
CHECK_SIZE_AND_OFFSET(msghdr, msg_control);
CHECK_SIZE_AND_OFFSET(msghdr, msg_controllen);
CHECK_SIZE_AND_OFFSET(msghdr, msg_flags);

CHECK_TYPE_SIZE(cmsghdr);
CHECK_SIZE_AND_OFFSET(cmsghdr, cmsg_len);
CHECK_SIZE_AND_OFFSET(cmsghdr, cmsg_level);
CHECK_SIZE_AND_OFFSET(cmsghdr, cmsg_type);

COMPILER_CHECK(sizeof(__sanitizer_dirent) <= sizeof(dirent));
CHECK_SIZE_AND_OFFSET(dirent, d_ino);
#if SANITIZER_MAC
CHECK_SIZE_AND_OFFSET(dirent, d_seekoff);
#else
CHECK_SIZE_AND_OFFSET(dirent, d_off);
#endif
CHECK_SIZE_AND_OFFSET(dirent, d_reclen);

#if SANITIZER_LINUX && !SANITIZER_ANDROID
COMPILER_CHECK(sizeof(__sanitizer_dirent64) <= sizeof(dirent64));
CHECK_SIZE_AND_OFFSET(dirent64, d_ino);
CHECK_SIZE_AND_OFFSET(dirent64, d_off);
CHECK_SIZE_AND_OFFSET(dirent64, d_reclen);
#endif

CHECK_TYPE_SIZE(ifconf);
CHECK_SIZE_AND_OFFSET(ifconf, ifc_len);
CHECK_SIZE_AND_OFFSET(ifconf, ifc_ifcu);

CHECK_TYPE_SIZE(pollfd);
CHECK_SIZE_AND_OFFSET(pollfd, fd);
CHECK_SIZE_AND_OFFSET(pollfd, events);
CHECK_SIZE_AND_OFFSET(pollfd, revents);
CHECK_TYPE_SIZE(nfds_t);

#endif  // SANITIZER_LINUX || SANITIZER_MAC

