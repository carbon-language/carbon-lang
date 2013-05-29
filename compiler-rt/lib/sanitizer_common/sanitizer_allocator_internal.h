//===-- sanitizer_allocator_internal.h -------------------------- C++ -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This allocator is used inside run-times.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_ALLOCATOR_INTERNAL_H
#define SANITIZER_ALLOCATOR_INTERNAL_H

#include "sanitizer_allocator.h"
#include "sanitizer_internal_defs.h"

namespace __sanitizer {

// FIXME: Check if we may use even more compact size class map for internal
// purposes.
typedef CompactSizeClassMap InternalSizeClassMap;

static const uptr kInternalAllocatorSpace = 0;
#if SANITIZER_WORDSIZE == 32
static const u64 kInternalAllocatorSize = (1ULL << 32);
static const uptr kInternalAllocatorRegionSizeLog = 20;
#else
static const u64 kInternalAllocatorSize = (1ULL << 47);
static const uptr kInternalAllocatorRegionSizeLog = 24;
#endif
static const uptr kInternalAllocatorFlatByteMapSize =
    kInternalAllocatorSize >> kInternalAllocatorRegionSizeLog;
typedef SizeClassAllocator32<
    kInternalAllocatorSpace, kInternalAllocatorSize, 16, InternalSizeClassMap,
    kInternalAllocatorRegionSizeLog,
    FlatByteMap<kInternalAllocatorFlatByteMapSize> > PrimaryInternalAllocator;

typedef SizeClassAllocatorLocalCache<PrimaryInternalAllocator>
    InternalAllocatorCache;

// We don't want our internal allocator to do any map/unmap operations.
struct CrashOnMapUnmap {
  void OnMap(uptr p, uptr size) const {
    RAW_CHECK_MSG(0, "Unexpected mmap in InternalAllocator!");
  }
  void OnUnmap(uptr p, uptr size) const {
    RAW_CHECK_MSG(0, "Unexpected munmap in InternalAllocator!");
  }
};

typedef CombinedAllocator<PrimaryInternalAllocator, InternalAllocatorCache,
                          LargeMmapAllocator<CrashOnMapUnmap> >
    InternalAllocator;

void *InternalAlloc(uptr size, InternalAllocatorCache *cache = 0);
void InternalFree(void *p, InternalAllocatorCache *cache = 0);
InternalAllocator *internal_allocator();

}  // namespace __sanitizer

#endif  // SANITIZER_ALLOCATOR_INTERNAL_H
