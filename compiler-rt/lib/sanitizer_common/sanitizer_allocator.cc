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
#ifdef __linux__
extern "C" void *__libc_malloc(__sanitizer::uptr size);
extern "C" void __libc_free(void *ptr);
# define LIBC_MALLOC __libc_malloc
# define LIBC_FREE __libc_free
#else  // __linux__
# include <stdlib.h>
# define LIBC_MALLOC malloc
# define LIBC_FREE free
#endif  // __linux__

namespace __sanitizer {

static const u64 kInternalAllocBlockMagic = 0x7A6CB03ABCEBC042ull;

void *InternalAlloc(uptr size) {
  void *p = LIBC_MALLOC(size + sizeof(u64));
  ((u64*)p)[0] = kInternalAllocBlockMagic;
  return (char*)p + sizeof(u64);
}

void InternalFree(void *addr) {
  if (!addr) return;
  addr = (char*)addr - sizeof(u64);
  CHECK_EQ(((u64*)addr)[0], kInternalAllocBlockMagic);
  ((u64*)addr)[0] = 0;
  LIBC_FREE(addr);
}

}  // namespace __sanitizer
