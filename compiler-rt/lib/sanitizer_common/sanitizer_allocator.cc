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

const u64 kBlockMagic = 0x6A6CB03ABCEBC041ull;

void *InternalAlloc(uptr size) {
  if (size + sizeof(u64) < size)
    return 0;
  void *p = LIBC_MALLOC(size + sizeof(u64));
  if (p == 0)
    return 0;
  ((u64*)p)[0] = kBlockMagic;
  return (char*)p + sizeof(u64);
}

void InternalFree(void *addr) {
  if (addr == 0)
    return;
  addr = (char*)addr - sizeof(u64);
  CHECK_EQ(((u64*)addr)[0], kBlockMagic);
  ((u64*)addr)[0] = 0;
  LIBC_FREE(addr);
}

void *InternalAllocBlock(void *p) {
  CHECK_NE(p, (void*)0);
  u64 *pp = (u64*)((uptr)p & ~0x7);
  for (; pp[0] != kBlockMagic; pp--) {}
  return pp + 1;
}

}  // namespace __sanitizer
