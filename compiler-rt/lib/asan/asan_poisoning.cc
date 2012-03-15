//===-- asan_poisoning.cc ---------------------------------------*- C++ -*-===//
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
#include "asan_interface.h"
#include "asan_internal.h"
#include "asan_mapping.h"

namespace __asan {

void PoisonShadow(uintptr_t addr, size_t size, uint8_t value) {
  CHECK(AddrIsAlignedByGranularity(addr));
  CHECK(AddrIsAlignedByGranularity(addr + size));
  uintptr_t shadow_beg = MemToShadow(addr);
  uintptr_t shadow_end = MemToShadow(addr + size);
  CHECK(REAL(memset) != NULL);
  REAL(memset)((void*)shadow_beg, value, shadow_end - shadow_beg);
}

void PoisonShadowPartialRightRedzone(uintptr_t addr,
                                     uintptr_t size,
                                     uintptr_t redzone_size,
                                     uint8_t value) {
  CHECK(AddrIsAlignedByGranularity(addr));
  uint8_t *shadow = (uint8_t*)MemToShadow(addr);
  for (uintptr_t i = 0; i < redzone_size;
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
  uint8_t *chunk;
  int8_t offset;  // in [0, SHADOW_GRANULARITY)
  int8_t value;  // = *chunk;

  explicit ShadowSegmentEndpoint(uintptr_t address) {
    chunk = (uint8_t*)MemToShadow(address);
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
void __asan_poison_memory_region(void const volatile *addr, size_t size) {
  if (!FLAG_allow_user_poisoning || size == 0) return;
  uintptr_t beg_addr = (uintptr_t)addr;
  uintptr_t end_addr = beg_addr + size;
  if (FLAG_v >= 1) {
    Printf("Trying to poison memory region [%p, %p)\n", beg_addr, end_addr);
  }
  ShadowSegmentEndpoint beg(beg_addr);
  ShadowSegmentEndpoint end(end_addr);
  if (beg.chunk == end.chunk) {
    CHECK(beg.offset < end.offset);
    int8_t value = beg.value;
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

void __asan_unpoison_memory_region(void const volatile *addr, size_t size) {
  if (!FLAG_allow_user_poisoning || size == 0) return;
  uintptr_t beg_addr = (uintptr_t)addr;
  uintptr_t end_addr = beg_addr + size;
  if (FLAG_v >= 1) {
    Printf("Trying to unpoison memory region [%p, %p)\n", beg_addr, end_addr);
  }
  ShadowSegmentEndpoint beg(beg_addr);
  ShadowSegmentEndpoint end(end_addr);
  if (beg.chunk == end.chunk) {
    CHECK(beg.offset < end.offset);
    int8_t value = beg.value;
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
  return __asan::AddressIsPoisoned((uintptr_t)addr);
}
