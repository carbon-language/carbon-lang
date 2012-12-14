//===-- asan_poisoning.cc -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Shadow memory poisoning by ASan RTL and by user application.
//===----------------------------------------------------------------------===//

#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "sanitizer/asan_interface.h"

namespace __asan {

void PoisonShadow(uptr addr, uptr size, u8 value) {
  CHECK(AddrIsAlignedByGranularity(addr));
  CHECK(AddrIsAlignedByGranularity(addr + size));
  uptr shadow_beg = MemToShadow(addr);
  uptr shadow_end = MemToShadow(addr + size);
  CHECK(REAL(memset) != 0);
  REAL(memset)((void*)shadow_beg, value, shadow_end - shadow_beg);
}

void PoisonShadowPartialRightRedzone(uptr addr,
                                     uptr size,
                                     uptr redzone_size,
                                     u8 value) {
  CHECK(AddrIsAlignedByGranularity(addr));
  u8 *shadow = (u8*)MemToShadow(addr);
  for (uptr i = 0; i < redzone_size;
       i += SHADOW_GRANULARITY, shadow++) {
    if (i + SHADOW_GRANULARITY <= size) {
      *shadow = 0;  // fully addressable
    } else if (i >= size) {
      *shadow = (SHADOW_GRANULARITY == 128) ? 0xff : value;  // unaddressable
    } else {
      *shadow = size - i;  // first size-i bytes are addressable
    }
  }
}


struct ShadowSegmentEndpoint {
  u8 *chunk;
  s8 offset;  // in [0, SHADOW_GRANULARITY)
  s8 value;  // = *chunk;

  explicit ShadowSegmentEndpoint(uptr address) {
    chunk = (u8*)MemToShadow(address);
    offset = address & (SHADOW_GRANULARITY - 1);
    value = *chunk;
  }
};

}  // namespace __asan

// ---------------------- Interface ---------------- {{{1
using namespace __asan;  // NOLINT

// Current implementation of __asan_(un)poison_memory_region doesn't check
// that user program (un)poisons the memory it owns. It poisons memory
// conservatively, and unpoisons progressively to make sure asan shadow
// mapping invariant is preserved (see detailed mapping description here:
// http://code.google.com/p/address-sanitizer/wiki/AddressSanitizerAlgorithm).
//
// * if user asks to poison region [left, right), the program poisons
// at least [left, AlignDown(right)).
// * if user asks to unpoison region [left, right), the program unpoisons
// at most [AlignDown(left), right).
void __asan_poison_memory_region(void const volatile *addr, uptr size) {
  if (!flags()->allow_user_poisoning || size == 0) return;
  uptr beg_addr = (uptr)addr;
  uptr end_addr = beg_addr + size;
  if (flags()->verbosity >= 1) {
    Printf("Trying to poison memory region [%p, %p)\n",
           (void*)beg_addr, (void*)end_addr);
  }
  ShadowSegmentEndpoint beg(beg_addr);
  ShadowSegmentEndpoint end(end_addr);
  if (beg.chunk == end.chunk) {
    CHECK(beg.offset < end.offset);
    s8 value = beg.value;
    CHECK(value == end.value);
    // We can only poison memory if the byte in end.offset is unaddressable.
    // No need to re-poison memory if it is poisoned already.
    if (value > 0 && value <= end.offset) {
      if (beg.offset > 0) {
        *beg.chunk = Min(value, beg.offset);
      } else {
        *beg.chunk = kAsanUserPoisonedMemoryMagic;
      }
    }
    return;
  }
  CHECK(beg.chunk < end.chunk);
  if (beg.offset > 0) {
    // Mark bytes from beg.offset as unaddressable.
    if (beg.value == 0) {
      *beg.chunk = beg.offset;
    } else {
      *beg.chunk = Min(beg.value, beg.offset);
    }
    beg.chunk++;
  }
  REAL(memset)(beg.chunk, kAsanUserPoisonedMemoryMagic, end.chunk - beg.chunk);
  // Poison if byte in end.offset is unaddressable.
  if (end.value > 0 && end.value <= end.offset) {
    *end.chunk = kAsanUserPoisonedMemoryMagic;
  }
}

void __asan_unpoison_memory_region(void const volatile *addr, uptr size) {
  if (!flags()->allow_user_poisoning || size == 0) return;
  uptr beg_addr = (uptr)addr;
  uptr end_addr = beg_addr + size;
  if (flags()->verbosity >= 1) {
    Printf("Trying to unpoison memory region [%p, %p)\n",
           (void*)beg_addr, (void*)end_addr);
  }
  ShadowSegmentEndpoint beg(beg_addr);
  ShadowSegmentEndpoint end(end_addr);
  if (beg.chunk == end.chunk) {
    CHECK(beg.offset < end.offset);
    s8 value = beg.value;
    CHECK(value == end.value);
    // We unpoison memory bytes up to enbytes up to end.offset if it is not
    // unpoisoned already.
    if (value != 0) {
      *beg.chunk = Max(value, end.offset);
    }
    return;
  }
  CHECK(beg.chunk < end.chunk);
  if (beg.offset > 0) {
    *beg.chunk = 0;
    beg.chunk++;
  }
  REAL(memset)(beg.chunk, 0, end.chunk - beg.chunk);
  if (end.offset > 0 && end.value != 0) {
    *end.chunk = Max(end.value, end.offset);
  }
}

bool __asan_address_is_poisoned(void const volatile *addr) {
  return __asan::AddressIsPoisoned((uptr)addr);
}

// This is a simplified version of __asan_(un)poison_memory_region, which
// assumes that left border of region to be poisoned is properly aligned.
static void PoisonAlignedStackMemory(uptr addr, uptr size, bool do_poison) {
  if (size == 0) return;
  uptr aligned_size = size & ~(SHADOW_GRANULARITY - 1);
  PoisonShadow(addr, aligned_size,
               do_poison ? kAsanStackUseAfterScopeMagic : 0);
  if (size == aligned_size)
    return;
  s8 end_offset = (s8)(size - aligned_size);
  s8* shadow_end = (s8*)MemToShadow(addr + aligned_size);
  s8 end_value = *shadow_end;
  if (do_poison) {
    // If possible, mark all the bytes mapping to last shadow byte as
    // unaddressable.
    if (end_value > 0 && end_value <= end_offset)
      *shadow_end = (s8)kAsanStackUseAfterScopeMagic;
  } else {
    // If necessary, mark few first bytes mapping to last shadow byte
    // as addressable
    if (end_value != 0)
      *shadow_end = Max(end_value, end_offset);
  }
}

void __asan_poison_stack_memory(uptr addr, uptr size) {
  if (flags()->verbosity > 0)
    Report("poisoning: %p %zx\n", (void*)addr, size);
  PoisonAlignedStackMemory(addr, size, true);
}

void __asan_unpoison_stack_memory(uptr addr, uptr size) {
  if (flags()->verbosity > 0)
    Report("unpoisoning: %p %zx\n", (void*)addr, size);
  PoisonAlignedStackMemory(addr, size, false);
}
