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

#include "sanitizer_internal_defs.h"

struct sigaltstack;

namespace __sanitizer {
// Dirent structure for getdents(). Note that this structure is different from
// the one in <dirent.h>, which is used by readdir().
struct linux_dirent;

// Syscall wrappers.
int internal_getdents(fd_t fd, struct linux_dirent *dirp, unsigned int count);
int internal_prctl(int option, uptr arg2, uptr arg3, uptr arg4, uptr arg5);
int internal_sigaltstack(const struct sigaltstack *ss, struct sigaltstack *oss);

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
  char buffer_[4096];
  bool error_;
  struct linux_dirent* entry_;
  int bytes_read_;
};

void AdjustStackSizeLinux(void *attr, int verbosity);

}  // namespace __sanitizer

#endif  // SANITIZER_LINUX_H
