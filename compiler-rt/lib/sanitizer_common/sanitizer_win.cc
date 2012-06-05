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

fd_t internal_open(const char *filename, bool write) {
  UNIMPLEMENTED_WIN();
  return 0;
}

}  // namespace __sanitizer

#endif  // _WIN32
