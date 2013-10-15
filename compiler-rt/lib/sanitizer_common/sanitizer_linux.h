//===-- sanitizer_linux.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Linux-specific syscall wrappers and classes.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_LINUX_H
#define SANITIZER_LINUX_H

#include "sanitizer_platform.h"
#if SANITIZER_LINUX
#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"

struct link_map;  // Opaque type returned by dlopen().
struct sigaltstack;
typedef struct siginfo siginfo_t;

namespace __sanitizer {
// Dirent structure for getdents(). Note that this structure is different from
// the one in <dirent.h>, which is used by readdir().
struct linux_dirent;

struct kernel_sigset_t {
  u8 sig[FIRST_32_SECOND_64(16, 8)];
};

struct kernel_sigaction_t {
  union {
    void (*sigaction)(int signo, siginfo_t *info, void *ctx);
    void (*handler)(int signo);
  };
  unsigned long sa_flags;
  void (*sa_restorer)(void);
  kernel_sigset_t sa_mask;
};

// Syscall wrappers.
uptr internal_getdents(fd_t fd, struct linux_dirent *dirp, unsigned int count);
uptr internal_prctl(int option, uptr arg2, uptr arg3, uptr arg4, uptr arg5);
uptr internal_sigaltstack(const struct sigaltstack* ss,
                          struct sigaltstack* oss);
uptr internal_sigaction(int signum, const kernel_sigaction_t *act,
    kernel_sigaction_t *oldact);
uptr internal_sigprocmask(int how, kernel_sigset_t *set,
    kernel_sigset_t *oldset);
void internal_sigfillset(kernel_sigset_t *set);
void internal_sigdelset(kernel_sigset_t *set, int signum);

#ifdef __x86_64__
uptr internal_clone(int (*fn)(void *), void *child_stack, int flags, void *arg,
                    int *parent_tidptr, void *newtls, int *child_tidptr);
#endif

// This class reads thread IDs from /proc/<pid>/task using only syscalls.
class ThreadLister {
 public:
  explicit ThreadLister(int pid);
  ~ThreadLister();
  // GetNextTID returns -1 if the list of threads is exhausted, or if there has
  // been an error.
  int GetNextTID();
  void Reset();
  bool error();

 private:
  bool GetDirectoryEntries();

  int pid_;
  int descriptor_;
  InternalScopedBuffer<char> buffer_;
  bool error_;
  struct linux_dirent* entry_;
  int bytes_read_;
};

void AdjustStackSizeLinux(void *attr, int verbosity);

// Exposed for testing.
uptr ThreadDescriptorSize();
uptr ThreadSelf();
uptr ThreadSelfOffset();

// Matches a library's file name against a base name (stripping path and version
// information).
bool LibraryNameIs(const char *full_name, const char *base_name);

// Read the name of the current binary from /proc/self/exe.
uptr ReadBinaryName(/*out*/char *buf, uptr buf_len);
// Cache the value of /proc/self/exe.
void CacheBinaryName();

// Call cb for each region mapped by map.
void ForEachMappedRegion(link_map *map, void (*cb)(const void *, uptr));
}  // namespace __sanitizer

#endif  // SANITIZER_LINUX
#endif  // SANITIZER_LINUX_H
