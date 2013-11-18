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

#include "asan_poisoning.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_flags.h"

namespace __asan {

void PoisonShadow(uptr addr, uptr size, u8 value) {
  if (!flags()->poison_heap) return;
  CHECK(AddrIsAlignedByGranularity(addr));
  CHECK(AddrIsInMem(addr));
  CHECK(AddrIsAlignedByGranularity(addr + size));
  CHECK(AddrIsInMem(addr + size - SHADOW_GRANULARITY));
  CHECK(REAL(memset));
  FastPoisonShadow(addr, size, value);
}

void PoisonShadowPartialRightRedzone(uptr addr,
                                     uptr size,
                                     uptr redzone_size,
                                     u8 value) {
  if (!flags()->poison_heap) return;
  CHECK(AddrIsAlignedByGranularity(addr));
  CHECK(AddrIsInMem(addr));
  FastPoisonShadowPartialRightRedzone(addr, size, redzone_size, value);
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
  if (common_flags()->verbosity >= 1) {
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
  if (common_flags()->verbosity >= 1) {
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

uptr __asan_region_is_poisoned(uptr beg, uptr size) {
  if (!size) return 0;
  uptr end = beg + size;
  if (!AddrIsInMem(beg)) return beg;
  if (!AddrIsInMem(end)) return end;
  uptr aligned_b = RoundUpTo(beg, SHADOW_GRANULARITY);
  uptr aligned_e = RoundDownTo(end, SHADOW_GRANULARITY);
  uptr shadow_beg = MemToShadow(aligned_b);
  uptr shadow_end = MemToShadow(aligned_e);
  // First check the first and the last application bytes,
  // then check the SHADOW_GRANULARITY-aligned region by calling
  // mem_is_zero on the corresponding shadow.
  if (!__asan::AddressIsPoisoned(beg) &&
      !__asan::AddressIsPoisoned(end - 1) &&
      (shadow_end <= shadow_beg ||
       __sanitizer::mem_is_zero((const char *)shadow_beg,
                                shadow_end - shadow_beg)))
    return 0;
  // The fast check failed, so we have a poisoned byte somewhere.
  // Find it slowly.
  for (; beg < end; beg++)
    if (__asan::AddressIsPoisoned(beg))
      return beg;
  UNREACHABLE("mem_is_zero returned false, but poisoned byte was not found");
  return 0;
}

#define CHECK_SMALL_REGION(p, size, isWrite)                  \
  do {                                                        \
    uptr __p = reinterpret_cast<uptr>(p);                     \
    uptr __size = size;                                       \
    if (UNLIKELY(__asan::AddressIsPoisoned(__p) ||            \
        __asan::AddressIsPoisoned(__p + __size - 1))) {       \
      GET_CURRENT_PC_BP_SP;                                   \
      uptr __bad = __asan_region_is_poisoned(__p, __size);    \
      __asan_report_error(pc, bp, sp, __bad, isWrite, __size);\
    }                                                         \
  } while (false);                                            \


extern "C" SANITIZER_INTERFACE_ATTRIBUTE
u16 __sanitizer_unaligned_load16(const uu16 *p) {
  CHECK_SMALL_REGION(p, sizeof(*p), false);
  return *p;
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE
u32 __sanitizer_unaligned_load32(const uu32 *p) {
  CHECK_SMALL_REGION(p, sizeof(*p), false);
  return *p;
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE
u64 __sanitizer_unaligned_load64(const uu64 *p) {
  CHECK_SMALL_REGION(p, sizeof(*p), false);
  return *p;
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_unaligned_store16(uu16 *p, u16 x) {
  CHECK_SMALL_REGION(p, sizeof(*p), true);
  *p = x;
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_unaligned_store32(uu32 *p, u32 x) {
  CHECK_SMALL_REGION(p, sizeof(*p), true);
  *p = x;
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_unaligned_store64(uu64 *p, u64 x) {
  CHECK_SMALL_REGION(p, sizeof(*p), true);
  *p = x;
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
  if (common_flags()->verbosity > 0)
    Report("poisoning: %p %zx\n", (void*)addr, size);
  PoisonAlignedStackMemory(addr, size, true);
}

void __asan_unpoison_stack_memory(uptr addr, uptr size) {
  if (common_flags()->verbosity > 0)
    Report("unpoisoning: %p %zx\n", (void*)addr, size);
  PoisonAlignedStackMemory(addr, size, false);
}

void __sanitizer_annotate_contiguous_container(void *beg_p, void *end_p,
                                               void *old_mid_p,
                                               void *new_mid_p) {
  uptr beg = reinterpret_cast<uptr>(beg_p);
  uptr end= reinterpret_cast<uptr>(end_p);
  uptr old_mid = reinterpret_cast<uptr>(old_mid_p);
  uptr new_mid = reinterpret_cast<uptr>(new_mid_p);
  uptr granularity = SHADOW_GRANULARITY;
  CHECK(beg <= end && beg <= old_mid && beg <= new_mid && old_mid <= end &&
        new_mid <= end && IsAligned(beg, granularity));
  CHECK_LE(end - beg,
           FIRST_32_SECOND_64(1UL << 30, 1UL << 34)); // Sanity check.

  uptr a = RoundDownTo(Min(old_mid, new_mid), granularity);
  uptr c = RoundUpTo(Max(old_mid, new_mid), granularity);
  uptr b = new_mid;
  uptr b1 = RoundDownTo(b, granularity);
  uptr b2 = RoundUpTo(b, granularity);
  uptr d = old_mid;
  uptr d1 = RoundDownTo(d, granularity);
  uptr d2 = RoundUpTo(d, granularity);
  // Currently we should be in this state:
  // [a, d1) is good, [d2, c) is bad, [d1, d2) is partially good.
  // Make a quick sanity check that we are indeed in this state.
  if (d1 != d2)
    CHECK_EQ(*(u8*)MemToShadow(d1), d - d1);
  if (a + granularity <= d1)
    CHECK_EQ(*(u8*)MemToShadow(a), 0);
  if (d2 + granularity <= c && c <= end)
    CHECK_EQ(*(u8 *)MemToShadow(c - granularity), kAsanUserPoisonedMemoryMagic);

  // New state:
  // [a, b1) is good, [b2, c) is bad, [b1, b2) is partially good.
  // FIXME: we may want to have a separate poison magic value.
  PoisonShadow(a, b1 - a, 0);
  PoisonShadow(b2, c - b2, kAsanUserPoisonedMemoryMagic);
  if (b1 != b2) {
    CHECK_EQ(b2 - b1, granularity);
    *(u8*)MemToShadow(b1) = b - b1;
  }
}
