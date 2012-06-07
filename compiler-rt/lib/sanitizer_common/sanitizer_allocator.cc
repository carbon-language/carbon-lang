//===-- sanitizer_allocator.cc --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
// This allocator that is used inside run-times.
//===----------------------------------------------------------------------===//
#include "sanitizer_common.h"

// FIXME: We should probably use more low-level allocator that would
// mmap some pages and split them into chunks to fulfill requests.
#include <stdlib.h>

namespace __sanitizer {

static const u64 kInternalAllocBlockMagic = 0x7A6CB03ABCEBC042ull;

void *InternalAlloc(uptr size) {
  void *p = malloc(size + sizeof(u64));
  ((u64*)p)[0] = kInternalAllocBlockMagic;
  return (char*)p + sizeof(u64);
}

void InternalFree(void *addr) {
  if (!addr) return;
  addr = (char*)addr - sizeof(u64);
  CHECK_EQ(((u64*)addr)[0], kInternalAllocBlockMagic);
  ((u64*)addr)[0] = 0;
  free(addr);
}

}  // namespace __sanitizer
