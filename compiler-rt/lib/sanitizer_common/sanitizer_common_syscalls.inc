//===-- sanitizer_common_syscalls.inc ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Common syscalls handlers for tools like AddressSanitizer,
// ThreadSanitizer, MemorySanitizer, etc.
//
// This file should be included into the tool's interceptor file,
// which has to define it's own macros:
//   COMMON_SYSCALL_PRE_READ_RANGE
//          Called in prehook for regions that will be read by the kernel and
//          must be initialized.
//   COMMON_SYSCALL_PRE_WRITE_RANGE
//          Called in prehook for regions that will be written to by the kernel
//          and must be addressable. The actual write range may be smaller than
//          reported in the prehook. See POST_WRITE_RANGE.
//   COMMON_SYSCALL_POST_READ_RANGE
//          Called in posthook for regions that were read by the kernel. Does
//          not make much sense.
//   COMMON_SYSCALL_POST_WRITE_RANGE
//          Called in posthook for regions that were written to by the kernel
//          and are now initialized.
//===----------------------------------------------------------------------===//

#define PRE_SYSCALL(name)                                                      \
  SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_syscall_pre_##name
#define PRE_READ(p, s) COMMON_SYSCALL_PRE_READ_RANGE(p, s)
#define PRE_WRITE(p, s) COMMON_SYSCALL_PRE_WRITE_RANGE(p, s)

#define POST_SYSCALL(name)                                                     \
  SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_syscall_post_##name
#define POST_READ(p, s) COMMON_SYSCALL_POST_READ_RANGE(p, s)
#define POST_WRITE(p, s) COMMON_SYSCALL_POST_WRITE_RANGE(p, s)

// FIXME: do some kind of PRE_READ for all syscall arguments (int(s) and such).

extern "C" {
struct sanitizer_kernel_iovec {
  void *iov_base;
  unsigned long iov_len;
};

struct sanitizer_kernel_msghdr {
  void *msg_name;
  int msg_namelen;
  struct sanitizer_kernel_iovec *msg_iov;
  unsigned long msg_iovlen;
  void *msg_control;
  unsigned long msg_controllen;
  unsigned msg_flags;
};

struct sanitizer_kernel_timespec {
  long tv_sec;
  long tv_nsec;
};

struct sanitizer_kernel_timeval {
  long tv_sec;
  long tv_usec;
};

struct sanitizer_kernel_rusage {
  struct sanitizer_kernel_timeval ru_timeval[2];
  long ru_long[14];
};

PRE_SYSCALL(recvmsg)(int sockfd, struct sanitizer_kernel_msghdr *msg,
                     int flags) {
  PRE_READ(msg, sizeof(*msg));
}

POST_SYSCALL(recvmsg)(long res, int sockfd, struct sanitizer_kernel_msghdr *msg,
                      int flags) {
  if (res > 0)
    for (unsigned long i = 0; i < msg->msg_iovlen; ++i) {
      POST_WRITE(msg->msg_iov[i].iov_base, msg->msg_iov[i].iov_len);
    }
  POST_WRITE(msg->msg_control, msg->msg_controllen);
}

PRE_SYSCALL(rt_sigpending)(void *p, unsigned long s) { PRE_WRITE(p, s); }

POST_SYSCALL(rt_sigpending)(long res, void *p, unsigned long s) {
  if (res == 0) {
    POST_WRITE(p, s);
  }
}

PRE_SYSCALL(getdents)(int fd, void *dirp, int count) { PRE_WRITE(dirp, count); }

POST_SYSCALL(getdents)(long res, int fd, void *dirp, int count) {
  if (res > 0) {
    POST_WRITE(dirp, res);
  }
}

PRE_SYSCALL(getdents64)(int fd, void *dirp, int count) {
  PRE_WRITE(dirp, count);
}

POST_SYSCALL(getdents64)(long res, int fd, void *dirp, int count) {
  if (res > 0) {
    POST_WRITE(dirp, res);
  }
}

PRE_SYSCALL(wait4)(int pid, int *status, int options,
                   struct sanitizer_kernel_rusage *r) {
  if (status) {
    PRE_WRITE(status, sizeof(*status));
  }
  if (r) {
    PRE_WRITE(r, sizeof(*r));
  }
}

POST_SYSCALL(wait4)(long res, int pid, int *status, int options,
                    struct sanitizer_kernel_rusage *r) {
  if (res > 0) {
    if (status) {
      POST_WRITE(status, sizeof(*status));
    }
    if (r) {
      POST_WRITE(r, sizeof(*r));
    }
  }
}

PRE_SYSCALL(waitpid)(int pid, int *status, int options) {
  if (status) {
    PRE_WRITE(status, sizeof(*status));
  }
}

POST_SYSCALL(waitpid)(long res, int pid, int *status, int options) {
  if (res > 0 && status) {
    POST_WRITE(status, sizeof(*status));
  }
}

PRE_SYSCALL(clock_gettime)(int clk_id, struct sanitizer_kernel_timespec *tp) {
  if (tp) PRE_WRITE(tp, sizeof(*tp));
}

POST_SYSCALL(clock_gettime)(long res, int clk_id, struct sanitizer_kernel_timespec *tp) {
  if (res == 0 && tp) POST_WRITE(tp, sizeof(*tp));
}

PRE_SYSCALL(clock_getres)(int clk_id, struct sanitizer_kernel_timespec *tp) {
  if (tp) PRE_WRITE(tp, sizeof(*tp));
}

POST_SYSCALL(clock_getres)(long res, int clk_id, struct sanitizer_kernel_timespec *tp) {
  if (res == 0 && tp) POST_WRITE(tp, sizeof(*tp));
}

}  // extern "C"

#undef PRE_SYSCALL
#undef PRE_READ
#undef PRE_WRITE
#undef POST_SYSCALL
#undef POST_READ
#undef POST_WRITE
