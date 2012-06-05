//===-- sanitizer_linux.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements linux-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//
#ifdef __linux__

#include "sanitizer_defs.h"
#include "sanitizer_libc.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

namespace __sanitizer {

void *internal_mmap(void *addr, uptr length, int prot, int flags,
                    int fd, u64 offset) {
#if __WORDSIZE == 64
  return (void *)syscall(__NR_mmap, addr, length, prot, flags, fd, offset);
#else
  return (void *)syscall(__NR_mmap2, addr, length, prot, flags, fd, offset);
#endif
}

int internal_close(fd_t fd) {
  return syscall(__NR_close, fd);
}

fd_t internal_open(const char *filename, bool write) {
  return syscall(__NR_open, filename,
      write ? O_WRONLY | O_CREAT | O_CLOEXEC : O_RDONLY, 0660);
}

uptr internal_read(fd_t fd, void *buf, uptr count) {
  return (uptr)syscall(__NR_read, fd, buf, count);
}

uptr internal_write(fd_t fd, const void *buf, uptr count) {
  return (uptr)syscall(__NR_write, fd, buf, count);
}

}  // namespace __sanitizer

#endif  // __linux__
