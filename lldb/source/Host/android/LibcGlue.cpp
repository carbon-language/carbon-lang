//===-- LibcGlue.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This files adds functions missing from libc on earlier versions of Android

#include <android/api-level.h>

#include <sys/syscall.h>
#include <lldb/Host/linux/Uio.h>

#if __ANDROID_API__ < 21

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <signal.h>

#include "lldb/Host/Time.h"

time_t timegm(struct tm* t)
{
    return (time_t) timegm64(t);
}

int signalfd (int fd, const sigset_t *mask, int flags)
{
    return syscall(__NR_signalfd4, fd, mask, _NSIG / 8, flags);
}

int posix_openpt(int flags)
{
    return open("/dev/ptmx", flags);
}

#endif

ssize_t process_vm_readv(::pid_t pid,
			 const struct iovec *local_iov, unsigned long liovcnt,
			 const struct iovec *remote_iov, unsigned long riovcnt,
			 unsigned long flags)
{
    return syscall(__NR_process_vm_readv, pid, local_iov, liovcnt, remote_iov, riovcnt, flags);
}
