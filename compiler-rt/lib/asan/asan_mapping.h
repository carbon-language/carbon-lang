//===-- asan_mapping.h ------------------------------------------*- C++ -*-===//
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
// Defines ASan memory mapping.
//===----------------------------------------------------------------------===//
#ifndef ASAN_MAPPING_H
#define ASAN_MAPPING_H

#include "asan_internal.h"

// The full explanation of the memory mapping could be found here:
// http://code.google.com/p/address-sanitizer/wiki/AddressSanitizerAlgorithm

#if ASAN_FLEXIBLE_MAPPING_AND_OFFSET == 1
extern __attribute__((visibility("default"))) uptr __asan_mapping_scale;
extern __attribute__((visibility("default"))) uptr __asan_mapping_offset;
# define SHADOW_SCALE (__asan_mapping_scale)
# define SHADOW_OFFSET (__asan_mapping_offset)
#else
# ifdef ANDROID
#  define SHADOW_SCALE (3)
#  define SHADOW_OFFSET (0)
# else
#  define SHADOW_SCALE (3)
#  if __WORDSIZE == 32
#   define SHADOW_OFFSET (1 << 29)
#  else
#   define SHADOW_OFFSET (1ULL << 44)
#  endif
# endif
#endif  // ASAN_FLEXIBLE_MAPPING_AND_OFFSET

#define SHADOW_GRANULARITY (1ULL << SHADOW_SCALE)
#define MEM_TO_SHADOW(mem) (((mem) >> SHADOW_SCALE) | (SHADOW_OFFSET))
#define SHADOW_TO_MEM(shadow) (((shadow) - SHADOW_OFFSET) << SHADOW_SCALE)

#if __WORDSIZE == 64
  static const uptr kHighMemEnd = 0x00007fffffffffffUL;
#else  // __WORDSIZE == 32
  static const uptr kHighMemEnd = 0xffffffff;
#endif  // __WORDSIZE


#define kLowMemBeg      0
#define kLowMemEnd      (SHADOW_OFFSET ? SHADOW_OFFSET - 1 : 0)

#define kLowShadowBeg   SHADOW_OFFSET
#define kLowShadowEnd   MEM_TO_SHADOW(kLowMemEnd)

#define kHighMemBeg     (MEM_TO_SHADOW(kHighMemEnd) + 1)

#define kHighShadowBeg  MEM_TO_SHADOW(kHighMemBeg)
#define kHighShadowEnd  MEM_TO_SHADOW(kHighMemEnd)

#define kShadowGapBeg   (kLowShadowEnd ? kLowShadowEnd + 1 : 16 * kPageSize)
#define kShadowGapEnd   (kHighShadowBeg - 1)

#define kGlobalAndStackRedzone \
      (SHADOW_GRANULARITY < 32 ? 32 : SHADOW_GRANULARITY)

namespace __asan {

static inline bool AddrIsInLowMem(uptr a) {
  return a < kLowMemEnd;
}

static inline bool AddrIsInLowShadow(uptr a) {
  return a >= kLowShadowBeg && a <= kLowShadowEnd;
}

static inline bool AddrIsInHighMem(uptr a) {
  return a >= kHighMemBeg && a <= kHighMemEnd;
}

static inline bool AddrIsInMem(uptr a) {
  return AddrIsInLowMem(a) || AddrIsInHighMem(a);
}

static inline uptr MemToShadow(uptr p) {
  CHECK(AddrIsInMem(p));
  return MEM_TO_SHADOW(p);
}

static inline bool AddrIsInHighShadow(uptr a) {
  return a >= kHighShadowBeg && a <=  kHighMemEnd;
}

static inline bool AddrIsInShadow(uptr a) {
  return AddrIsInLowShadow(a) || AddrIsInHighShadow(a);
}

static inline bool AddrIsInShadowGap(uptr a) {
  return a >= kShadowGapBeg && a <= kShadowGapEnd;
}

static inline bool AddrIsAlignedByGranularity(uptr a) {
  return (a & (SHADOW_GRANULARITY - 1)) == 0;
}

static inline bool AddressIsPoisoned(uptr a) {
  const uptr kAccessSize = 1;
  u8 *shadow_address = (u8*)MemToShadow(a);
  s8 shadow_value = *shadow_address;
  if (shadow_value) {
    u8 last_accessed_byte = (a & (SHADOW_GRANULARITY - 1))
                                 + kAccessSize - 1;
    return (last_accessed_byte >= shadow_value);
  }
  return false;
}

}  // namespace __asan

#endif  // ASAN_MAPPING_H
