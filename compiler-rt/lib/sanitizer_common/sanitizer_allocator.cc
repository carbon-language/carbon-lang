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
#if SANITIZER_LINUX && !SANITIZER_ANDROID
extern "C" void *__libc_malloc(__sanitizer::uptr size);
extern "C" void __libc_free(void *ptr);
# define LIBC_MALLOC __libc_malloc
# define LIBC_FREE __libc_free
#else  // __linux__ && !ANDROID
# include <stdlib.h>
# define LIBC_MALLOC malloc
# define LIBC_FREE free
#endif  // __linux__ && !ANDROID

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

// LowLevelAllocator
static LowLevelAllocateCallback low_level_alloc_callback;

void *LowLevelAllocator::Allocate(uptr size) {
  // Align allocation size.
  size = RoundUpTo(size, 8);
  if (allocated_end_ - allocated_current_ < (sptr)size) {
    uptr size_to_allocate = Max(size, GetPageSizeCached());
    allocated_current_ =
        (char*)MmapOrDie(size_to_allocate, __FUNCTION__);
    allocated_end_ = allocated_current_ + size_to_allocate;
    if (low_level_alloc_callback) {
      low_level_alloc_callback((uptr)allocated_current_,
                               size_to_allocate);
    }
  }
  CHECK(allocated_end_ - allocated_current_ >= (sptr)size);
  void *res = allocated_current_;
  allocated_current_ += size;
  return res;
}

void SetLowLevelAllocateCallback(LowLevelAllocateCallback callback) {
  low_level_alloc_callback = callback;
}

bool CallocShouldReturnNullDueToOverflow(uptr size, uptr n) {
  if (!size) return false;
  uptr max = (uptr)-1L;
  return (max / size) < n;
}

}  // namespace __sanitizer
