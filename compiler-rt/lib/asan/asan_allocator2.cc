//===-- asan_allocator2.cc ------------------------------------------------===//
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
// Implementation of ASan's memory allocator, 2-nd version.
// This variant uses the allocator from sanitizer_common, i.e. the one shared
// with ThreadSanitizer and MemorySanitizer.
//
// Status: under development, not enabled by default yet.
//===----------------------------------------------------------------------===//
#include "asan_allocator.h"
#if ASAN_ALLOCATOR_VERSION == 2

#include "sanitizer_common/sanitizer_allocator.h"

namespace __asan {

#if SANITIZER_WORDSIZE == 64
const uptr kAllocatorSpace = 0x600000000000ULL;
const uptr kAllocatorSize  =  0x10000000000ULL;  // 1T.
typedef SizeClassAllocator64<kAllocatorSpace, kAllocatorSize, 0 /*metadata*/,
    DefaultSizeClassMap> PrimaryAllocator;
#elif SANITIZER_WORDSIZE == 32
static const u64 kAddressSpaceSize = 1ULL << 32;
typedef SizeClassAllocator32<
  0, kAddressSpaceSize, 16, CompactSizeClassMap> PrimaryAllocator;
#endif

typedef SizeClassAllocatorLocalCache<PrimaryAllocator> AllocatorCache;
typedef LargeMmapAllocator SecondaryAllocator;
typedef CombinedAllocator<PrimaryAllocator, AllocatorCache,
    SecondaryAllocator> Allocator;


}  // namespace __asan
#endif  // ASAN_ALLOCATOR_VERSION
