//===-- sanitizer_win.cc --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements windows-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//
#ifdef _WIN32
#include <windows.h>

#include <assert.h>

#include "sanitizer_defs.h"
#include "sanitizer_libc.h"

#define UNIMPLEMENTED_WIN() assert(false)

namespace __sanitizer {

void *internal_mmap(void *addr, uptr length, int prot, int flags,
                    int fd, u64 offset) {
  UNIMPLEMENTED_WIN();
  return 0;
}

int internal_munmap(void *addr, uptr length) {
  UNIMPLEMENTED_WIN();
  return 0;
}

int internal_close(fd_t fd) {
  UNIMPLEMENTED_WIN();
  return 0;
}

fd_t internal_open(const char *filename, bool write) {
  UNIMPLEMENTED_WIN();
  return 0;
}

uptr internal_read(fd_t fd, void *buf, uptr count) {
  UNIMPLEMENTED_WIN();
  return 0;
}

uptr internal_write(fd_t fd, const void *buf, uptr count) {
  if (fd != 2) {
    UNIMPLEMENTED_WIN();
    return 0;
  }
  HANDLE err = GetStdHandle(STD_ERROR_HANDLE);
  if (err == 0)
    return 0;  // FIXME: this might not work on some apps.
  DWORD ret;
  if (!WriteFile(err, buf, count, &ret, 0))
    return 0;
  return ret;
}

int internal_sscanf(const char *str, const char *format, ...) {
  UNIMPLEMENTED_WIN();
  return -1;
}

}  // namespace __sanitizer

#endif  // _WIN32
