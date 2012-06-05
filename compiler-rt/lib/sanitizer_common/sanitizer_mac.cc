//===-- sanitizer_mac.cc --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements mac-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//

#ifdef __APPLE__

#include "sanitizer_defs.h"
#include "sanitizer_libc.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

namespace __sanitizer {

void *internal_mmap(void *addr, size_t length, int prot, int flags,
                    int fd, u64 offset) {
  return mmap(addr, length, prot, flags, fd, offset);
}

int internal_munmap(void *addr, uptr length) {
  return munmap(addr, length);
}

int internal_close(fd_t fd) {
  return close(fd);
}

fd_t internal_open(const char *filename, bool write) {
  return open(filename,
              write ? O_WRONLY | O_CREAT : O_RDONLY, 0660);
}

uptr internal_read(fd_t fd, void *buf, uptr count) {
  return read(fd, buf, count);
}

uptr internal_write(fd_t fd, const void *buf, uptr count) {
  return write(fd, buf, count);
}

}  // namespace __sanitizer

#endif  // __APPLE__
