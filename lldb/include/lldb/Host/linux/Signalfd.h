//===-- Signalfd.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This file defines signalfd functions & structures

#ifndef liblldb_Host_linux_Signalfd_h_
#define liblldb_Host_linux_Signalfd_h_

#ifdef __ANDROID_NDK__
#include <android/api-level.h>
#endif

#if defined(__ANDROID_API__) && __ANDROID_API__ < 21

#include <linux/types.h>
#include <linux/fcntl.h>

#define SFD_CLOEXEC O_CLOEXEC
#define SFD_NONBLOCK O_NONBLOCK

struct signalfd_siginfo {
    __u32 ssi_signo;
    __s32 ssi_errno;
    __s32 ssi_code;
    __u32 ssi_pid;
    __u32 ssi_uid;
    __s32 ssi_fd;
    __u32 ssi_tid;
    __u32 ssi_band;
    __u32 ssi_overrun;
    __u32 ssi_trapno;
    __s32 ssi_status;
    __s32 ssi_int;
    __u64 ssi_ptr;
    __u64 ssi_utime;
    __u64 ssi_stime;
    __u64 ssi_addr;
    __u16 ssi_addr_lsb;
    __u8 __pad[46];
};

int signalfd (int fd, const sigset_t *mask, int flags);

#else
#include <sys/signalfd.h>
#endif

#endif // liblldb_Host_linux_Signalfd_h_
