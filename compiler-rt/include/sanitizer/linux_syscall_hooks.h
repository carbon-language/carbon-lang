//===-- linux_syscall_hooks.h ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of public sanitizer interface.
//
// System call handlers.
//
// Interface methods declared in this header implement pre- and post- syscall
// actions for the active sanitizer.
// Usage:
//   __sanitizer_syscall_pre_getfoo(...args...);
//   int res = syscall(__NR_getfoo, ...args...);
//   __sanitizer_syscall_post_getfoo(res, ...args...);
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_LINUX_SYSCALL_HOOKS_H
#define SANITIZER_LINUX_SYSCALL_HOOKS_H

#ifdef __cplusplus
extern "C" {
#endif

void __sanitizer_syscall_pre_rt_sigpending(void *p, size_t s);
void __sanitizer_syscall_pre_getdents(int fd, void *dirp, int count);
void __sanitizer_syscall_pre_getdents64(int fd, void *dirp, int count);
void __sanitizer_syscall_pre_recvmsg(int sockfd, void *msg, int flags);

void __sanitizer_syscall_post_rt_sigpending(int res, void *p, size_t s);
void __sanitizer_syscall_post_getdents(int res, int fd, void *dirp, int count);
void __sanitizer_syscall_post_getdents64(int res, int fd, void *dirp, int count);
void __sanitizer_syscall_post_recvmsg(int res, int sockfd, void *msg, int flags);

// And now a few syscalls we don't handle yet.

#define __sanitizer_syscall_pre__gettid()
#define __sanitizer_syscall_pre_fork()
#define __sanitizer_syscall_pre_getegid()
#define __sanitizer_syscall_pre_geteuid()
#define __sanitizer_syscall_pre_getpgrp()
#define __sanitizer_syscall_pre_getpid()
#define __sanitizer_syscall_pre_getppid()
#define __sanitizer_syscall_pre_sched_yield()
#define __sanitizer_syscall_pre_setsid()

#define __sanitizer_syscall_pre__exit(a)
#define __sanitizer_syscall_pre_brk(a)
#define __sanitizer_syscall_pre_chdir(a)
#define __sanitizer_syscall_pre_chroot(a)
#define __sanitizer_syscall_pre_close(a)
#define __sanitizer_syscall_pre_dup(a)
#define __sanitizer_syscall_pre_exit_group(a)
#define __sanitizer_syscall_pre_getsid(a)
#define __sanitizer_syscall_pre_io_destroy(a)
#define __sanitizer_syscall_pre_pipe(a)
#define __sanitizer_syscall_pre_rt_sigreturn(a)
#define __sanitizer_syscall_pre_set_tid_address(a)
#define __sanitizer_syscall_pre_setfsgid(a)
#define __sanitizer_syscall_pre_setfsuid(a)
#define __sanitizer_syscall_pre_setgid(a)
#define __sanitizer_syscall_pre_setuid(a)
#define __sanitizer_syscall_pre_umask(a)
#define __sanitizer_syscall_pre_unlink(a)
#define __sanitizer_syscall_pre_unshare(a)

#define __sanitizer_syscall_pre_arch_prctl(a, b)
#define __sanitizer_syscall_pre_capset(a, b)
#define __sanitizer_syscall_pre_clock_getres(a, b)
#define __sanitizer_syscall_pre_clock_gettime(a, b)
#define __sanitizer_syscall_pre_dup2(a, b)
#define __sanitizer_syscall_pre_fstat(a, b)
#define __sanitizer_syscall_pre_fstatfs(a, b)
#define __sanitizer_syscall_pre_ftruncate(a, b)
#define __sanitizer_syscall_pre_getpriority(a, b)
#define __sanitizer_syscall_pre_getrlimit(a, b)
#define __sanitizer_syscall_pre_gettimeofday(a, b)
#define __sanitizer_syscall_pre_io_setup(a, b)
#define __sanitizer_syscall_pre_ioprio_get(a, b)
#define __sanitizer_syscall_pre_kill(a, b)
#define __sanitizer_syscall_pre_munmap(a, b)
#define __sanitizer_syscall_pre_prctl(a, b)
#define __sanitizer_syscall_pre_rt_sigsuspend(a, b)
#define __sanitizer_syscall_pre_setgroups(a, b)
#define __sanitizer_syscall_pre_setns(a, b)
#define __sanitizer_syscall_pre_setpgid(a, b)
#define __sanitizer_syscall_pre_setrlimit(a, b)
#define __sanitizer_syscall_pre_shutdown(a, b)
#define __sanitizer_syscall_pre_sigaltstack(a, b)
#define __sanitizer_syscall_pre_stat(a, b)
#define __sanitizer_syscall_pre_statfs(a, b)
#define __sanitizer_syscall_pre_tkill(a, b)

#define __sanitizer_syscall_pre_execve(a, b, c)
#define __sanitizer_syscall_pre_fcntl(a, b, c)
#define __sanitizer_syscall_pre_getcpu(a, b, c)
#define __sanitizer_syscall_pre_getresgid(a, b, c)
#define __sanitizer_syscall_pre_getresuid(a, b, c)
#define __sanitizer_syscall_pre_io_cancel(a, b, c)
#define __sanitizer_syscall_pre_io_submit(a, b, c)
#define __sanitizer_syscall_pre_ioctl(a, b, c)
#define __sanitizer_syscall_pre_ioprio_set(a, b, c)
#define __sanitizer_syscall_pre_listxattr(a, b, c)
#define __sanitizer_syscall_pre_llistxattr(a, b, c)
#define __sanitizer_syscall_pre_lseek(a, b, c)
#define __sanitizer_syscall_pre_mprotect(a, b, c)
#define __sanitizer_syscall_pre_open(a, b, c)
#define __sanitizer_syscall_pre_poll(a, b, c)
#define __sanitizer_syscall_pre_read(a, b, c)
#define __sanitizer_syscall_pre_readahead(a, b, c)
#define __sanitizer_syscall_pre_readlink(a, b, c)
#define __sanitizer_syscall_pre_sched_getaffinity(a, b, c)
#define __sanitizer_syscall_pre_sched_setaffinity(a, b, c)
#define __sanitizer_syscall_pre_sendmsg(a, b, c)
#define __sanitizer_syscall_pre_setpriority(a, b, c)
#define __sanitizer_syscall_pre_setresgid(a, b, c)
#define __sanitizer_syscall_pre_setresuid(a, b, c)
#define __sanitizer_syscall_pre_socket(a, b, c)
#define __sanitizer_syscall_pre_tgkill(a, b, c)
#define __sanitizer_syscall_pre_unlinkat(a, b, c)
#define __sanitizer_syscall_pre_write(a, b, c)
#define __sanitizer_syscall_pre_writev(a, b, c)

#define __sanitizer_syscall_pre_fadvise64(a, b, c, d)
#define __sanitizer_syscall_pre_fallocate(a, b, c, d)
#define __sanitizer_syscall_pre_futex(a, b, c, d)
#define __sanitizer_syscall_pre_getxattr(a, b, c, d)
#define __sanitizer_syscall_pre_lgetxattr(a, b, c, d)
#define __sanitizer_syscall_pre_newfstatat(a, b, c, d)
#define __sanitizer_syscall_pre_openat(a, b, c, d)
#define __sanitizer_syscall_pre_pread64(a, b, c, d)
#define __sanitizer_syscall_pre_ptrace(a, b, c, d)
#define __sanitizer_syscall_pre_pwrite64(a, b, c, d)
#define __sanitizer_syscall_pre_quotactl(a, b, c, d)
#define __sanitizer_syscall_pre_rt_sigaction(a, b, c, d)
#define __sanitizer_syscall_pre_rt_sigprocmask(a, b, c, d)
#define __sanitizer_syscall_pre_socketpair(a, b, c, d)
#define __sanitizer_syscall_pre_wait4(a, b, c, d)

#define __sanitizer_syscall_pre__mremap(a, b, c, d, e)
#define __sanitizer_syscall_pre_io_getevents(a, b, c, d, e)
#define __sanitizer_syscall_pre_lsetxattr(a, b, c, d, e)
#define __sanitizer_syscall_pre_mount(a, b, c, d, e)
#define __sanitizer_syscall_pre_preadv(a, b, c, d, e)
#define __sanitizer_syscall_pre_pwritev(a, b, c, d, e)
#define __sanitizer_syscall_pre_setxattr(a, b, c, d, e)

#define __sanitizer_syscall_pre_mmap(a, b, c, d, e, f)
#define __sanitizer_syscall_pre_move_pages(a, b, c, d, e, f)
#define __sanitizer_syscall_pre_sendto(a, b, c, d, e, f)


#define __sanitizer_syscall_post__gettid(res)
#define __sanitizer_syscall_post_fork(res)
#define __sanitizer_syscall_post_getegid(res)
#define __sanitizer_syscall_post_geteuid(res)
#define __sanitizer_syscall_post_getpgrp(res)
#define __sanitizer_syscall_post_getpid(res)
#define __sanitizer_syscall_post_getppid(res)
#define __sanitizer_syscall_post_sched_yield(res)
#define __sanitizer_syscall_post_setsid(res)

#define __sanitizer_syscall_post__exit(res, a)
#define __sanitizer_syscall_post_brk(res, a)
#define __sanitizer_syscall_post_chdir(res, a)
#define __sanitizer_syscall_post_chroot(res, a)
#define __sanitizer_syscall_post_close(res, a)
#define __sanitizer_syscall_post_dup(res, a)
#define __sanitizer_syscall_post_exit_group(res, a)
#define __sanitizer_syscall_post_getsid(res, a)
#define __sanitizer_syscall_post_io_destroy(res, a)
#define __sanitizer_syscall_post_pipe(res, a)
#define __sanitizer_syscall_post_rt_sigreturn(res, a)
#define __sanitizer_syscall_post_set_tid_address(res, a)
#define __sanitizer_syscall_post_setfsgid(res, a)
#define __sanitizer_syscall_post_setfsuid(res, a)
#define __sanitizer_syscall_post_setgid(res, a)
#define __sanitizer_syscall_post_setuid(res, a)
#define __sanitizer_syscall_post_umask(res, a)
#define __sanitizer_syscall_post_unlink(res, a)
#define __sanitizer_syscall_post_unshare(res, a)

#define __sanitizer_syscall_post_arch_prctl(res, a, b)
#define __sanitizer_syscall_post_capset(res, a, b)
#define __sanitizer_syscall_post_clock_getres(res, a, b)
#define __sanitizer_syscall_post_clock_gettime(res, a, b)
#define __sanitizer_syscall_post_dup2(res, a, b)
#define __sanitizer_syscall_post_fstat(res, a, b)
#define __sanitizer_syscall_post_fstatfs(res, a, b)
#define __sanitizer_syscall_post_ftruncate(res, a, b)
#define __sanitizer_syscall_post_getpriority(res, a, b)
#define __sanitizer_syscall_post_getrlimit(res, a, b)
#define __sanitizer_syscall_post_gettimeofday(res, a, b)
#define __sanitizer_syscall_post_io_setup(res, a, b)
#define __sanitizer_syscall_post_ioprio_get(res, a, b)
#define __sanitizer_syscall_post_kill(res, a, b)
#define __sanitizer_syscall_post_munmap(res, a, b)
#define __sanitizer_syscall_post_prctl(res, a, b)
#define __sanitizer_syscall_post_rt_sigsuspend(res, a, b)
#define __sanitizer_syscall_post_setgroups(res, a, b)
#define __sanitizer_syscall_post_setns(res, a, b)
#define __sanitizer_syscall_post_setpgid(res, a, b)
#define __sanitizer_syscall_post_setrlimit(res, a, b)
#define __sanitizer_syscall_post_shutdown(res, a, b)
#define __sanitizer_syscall_post_sigaltstack(res, a, b)
#define __sanitizer_syscall_post_stat(res, a, b)
#define __sanitizer_syscall_post_statfs(res, a, b)
#define __sanitizer_syscall_post_tkill(res, a, b)

#define __sanitizer_syscall_post_execve(res, a, b, c)
#define __sanitizer_syscall_post_fcntl(res, a, b, c)
#define __sanitizer_syscall_post_getcpu(res, a, b, c)
#define __sanitizer_syscall_post_getresgid(res, a, b, c)
#define __sanitizer_syscall_post_getresuid(res, a, b, c)
#define __sanitizer_syscall_post_io_cancel(res, a, b, c)
#define __sanitizer_syscall_post_io_submit(res, a, b, c)
#define __sanitizer_syscall_post_ioctl(res, a, b, c)
#define __sanitizer_syscall_post_ioprio_set(res, a, b, c)
#define __sanitizer_syscall_post_listxattr(res, a, b, c)
#define __sanitizer_syscall_post_llistxattr(res, a, b, c)
#define __sanitizer_syscall_post_lseek(res, a, b, c)
#define __sanitizer_syscall_post_mprotect(res, a, b, c)
#define __sanitizer_syscall_post_open(res, a, b, c)
#define __sanitizer_syscall_post_poll(res, a, b, c)
#define __sanitizer_syscall_post_read(res, a, b, c)
#define __sanitizer_syscall_post_readahead(res, a, b, c)
#define __sanitizer_syscall_post_readlink(res, a, b, c)
#define __sanitizer_syscall_post_sched_getaffinity(res, a, b, c)
#define __sanitizer_syscall_post_sched_setaffinity(res, a, b, c)
#define __sanitizer_syscall_post_sendmsg(res, a, b, c)
#define __sanitizer_syscall_post_setpriority(res, a, b, c)
#define __sanitizer_syscall_post_setresgid(res, a, b, c)
#define __sanitizer_syscall_post_setresuid(res, a, b, c)
#define __sanitizer_syscall_post_socket(res, a, b, c)
#define __sanitizer_syscall_post_tgkill(res, a, b, c)
#define __sanitizer_syscall_post_unlinkat(res, a, b, c)
#define __sanitizer_syscall_post_write(res, a, b, c)
#define __sanitizer_syscall_post_writev(res, a, b, c)

#define __sanitizer_syscall_post_fadvise64(res, a, b, c, d)
#define __sanitizer_syscall_post_fallocate(res, a, b, c, d)
#define __sanitizer_syscall_post_futex(res, a, b, c, d)
#define __sanitizer_syscall_post_getxattr(res, a, b, c, d)
#define __sanitizer_syscall_post_lgetxattr(res, a, b, c, d)
#define __sanitizer_syscall_post_newfstatat(res, a, b, c, d)
#define __sanitizer_syscall_post_openat(res, a, b, c, d)
#define __sanitizer_syscall_post_pread64(res, a, b, c, d)
#define __sanitizer_syscall_post_ptrace(res, a, b, c, d)
#define __sanitizer_syscall_post_pwrite64(res, a, b, c, d)
#define __sanitizer_syscall_post_quotactl(res, a, b, c, d)
#define __sanitizer_syscall_post_rt_sigaction(res, a, b, c, d)
#define __sanitizer_syscall_post_rt_sigprocmask(res, a, b, c, d)
#define __sanitizer_syscall_post_socketpair(res, a, b, c, d)
#define __sanitizer_syscall_post_wait4(res, a, b, c, d)

#define __sanitizer_syscall_post__mremap(res, a, b, c, d, e)
#define __sanitizer_syscall_post_io_getevents(res, a, b, c, d, e)
#define __sanitizer_syscall_post_lsetxattr(res, a, b, c, d, e)
#define __sanitizer_syscall_post_mount(res, a, b, c, d, e)
#define __sanitizer_syscall_post_preadv(res, a, b, c, d, e)
#define __sanitizer_syscall_post_pwritev(res, a, b, c, d, e)
#define __sanitizer_syscall_post_setxattr(res, a, b, c, d, e)

#define __sanitizer_syscall_post_mmap(res, a, b, c, d, e, f)
#define __sanitizer_syscall_post_move_pages(res, a, b, c, d, e, f)
#define __sanitizer_syscall_post_sendto(res, a, b, c, d, e, f)

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SANITIZER_LINUX_SYSCALL_HOOKS_H
