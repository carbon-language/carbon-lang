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

namespace __sanitizer {

void *internal_mmap(void *addr, size_t length, int prot, int flags,
                    int fd, u64 offset) {
  return mmap(addr, length, prot, flags, fd, offset);
}

}  // namespace __sanitizer

#endif  // __APPLE__
