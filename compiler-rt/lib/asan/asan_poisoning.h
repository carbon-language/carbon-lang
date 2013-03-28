//===-- asan_poisoning.h ----------------------------------------*- C++ -*-===//
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

namespace __asan {

// Poisons the shadow memory for "size" bytes starting from "addr".
void PoisonShadow(uptr addr, uptr size, u8 value);

// Poisons the shadow memory for "redzone_size" bytes starting from
// "addr + size".
void PoisonShadowPartialRightRedzone(uptr addr,
                                     uptr size,
                                     uptr redzone_size,
                                     u8 value);

// Fast versions of PoisonShadow and PoisonShadowPartialRightRedzone that
// assume that memory addresses are properly aligned. Use in
// performance-critical code with care.
ALWAYS_INLINE void FastPoisonShadow(uptr aligned_beg, uptr aligned_size,
                                    u8 value) {
  DCHECK(flags()->poison_heap);
  uptr shadow_beg = MEM_TO_SHADOW(aligned_beg);
  uptr shadow_end = MEM_TO_SHADOW(
      aligned_beg + aligned_size - SHADOW_GRANULARITY) + 1;
  REAL(memset)((void*)shadow_beg, value, shadow_end - shadow_beg);
}

ALWAYS_INLINE void FastPoisonShadowPartialRightRedzone(
    uptr aligned_addr, uptr size, uptr redzone_size, u8 value) {
  DCHECK(flags()->poison_heap);
  u8 *shadow = (u8*)MEM_TO_SHADOW(aligned_addr);
  for (uptr i = 0; i < redzone_size; i += SHADOW_GRANULARITY, shadow++) {
    if (i + SHADOW_GRANULARITY <= size) {
      *shadow = 0;  // fully addressable
    } else if (i >= size) {
      *shadow = (SHADOW_GRANULARITY == 128) ? 0xff : value;  // unaddressable
    } else {
      *shadow = size - i;  // first size-i bytes are addressable
    }
  }
}

}  // namespace __asan
