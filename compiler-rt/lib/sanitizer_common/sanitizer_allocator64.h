//===-- sanitizer_allocator64.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Specialized allocator which works only in 64-bit address space.
// To be used by ThreadSanitizer, MemorySanitizer and possibly other tools.
// The main feature of this allocator is that the header is located far away
// from the user memory region, so that the tool does not use extra shadow
// for the header.
//
// Status: not yet ready.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_ALLOCATOR_H
#define SANITIZER_ALLOCATOR_H

#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"

namespace __sanitizer {

// Maps size class id to size and back.
class DefaultSizeClassMap {
 private:
  // Here we use a spline composed of 5 polynomials of oder 1.
  // The first size class is l0, then the classes go with step s0
  // untill they reach l1, after which they go with step s1 and so on.
  // Steps should be powers of two for cheap division.
  // The size of the last size class should be a power of two.
  // There should be at most 256 size classes.
  static const uptr l0 = 1 << 4;
  static const uptr l1 = 1 << 9;
  static const uptr l2 = 1 << 12;
  static const uptr l3 = 1 << 15;
  static const uptr l4 = 1 << 18;
  static const uptr l5 = 1 << 21;

  static const uptr s0 = 1 << 4;
  static const uptr s1 = 1 << 6;
  static const uptr s2 = 1 << 9;
  static const uptr s3 = 1 << 12;
  static const uptr s4 = 1 << 15;

  static const uptr u0 = 0  + (l1 - l0) / s0;
  static const uptr u1 = u0 + (l2 - l1) / s1;
  static const uptr u2 = u1 + (l3 - l2) / s2;
  static const uptr u3 = u2 + (l4 - l3) / s3;
  static const uptr u4 = u3 + (l5 - l4) / s4;

 public:
  static const uptr kNumClasses = u4 + 1;
  static const uptr kMaxSize = l5;
  static const uptr kMinSize = l0;

  COMPILER_CHECK(kNumClasses <= 256);
  COMPILER_CHECK((kMaxSize & (kMaxSize - 1)) == 0);

  static uptr Size(uptr class_id) {
    if (class_id <= u0) return l0 + s0 * (class_id - 0);
    if (class_id <= u1) return l1 + s1 * (class_id - u0);
    if (class_id <= u2) return l2 + s2 * (class_id - u1);
    if (class_id <= u3) return l3 + s3 * (class_id - u2);
    if (class_id <= u4) return l4 + s4 * (class_id - u3);
    return 0;
  }
  static uptr ClassID(uptr size) {
    if (size <= l1) return 0  + (size - l0 + s0 - 1) / s0;
    if (size <= l2) return u0 + (size - l1 + s1 - 1) / s1;
    if (size <= l3) return u1 + (size - l2 + s2 - 1) / s2;
    if (size <= l4) return u2 + (size - l3 + s3 - 1) / s3;
    if (size <= l5) return u3 + (size - l4 + s4 - 1) / s4;
    return 0;
  }
};

// Space: a portion of address space of kSpaceSize bytes starting at
// a fixed address (kSpaceBeg). Both constants are powers of two and
// kSpaceBeg is kSpaceSize-aligned.
//
// Region: a part of Space dedicated to a single size class.
// There are kNumClasses Regions of equal size.
//
// UserChunk: a piece of memory returned to user.
// MetaChunk: kMetadataSize bytes of metadata associated with a UserChunk.
//
// A Region looks like this:
// UserChunk1 ... UserChunkN <gap> MetaChunkN ... MetaChunk1
template <const uptr kSpaceBeg, const uptr kSpaceSize,
          const uptr kMetadataSize, class SizeClassMap>
class SizeClassAllocator64 {
 public:
  void Init() {
    CHECK_EQ(AllocBeg(), reinterpret_cast<uptr>(MmapFixedNoReserve(
             AllocBeg(), AllocSize())));
  }
  void *Allocate(uptr size) {
    CHECK_LE(size, SizeClassMap::kMaxSize);
    return AllocateBySizeClass(SizeClassMap::ClassID(size));
  }
  void Deallocate(void *p) {
    DeallocateBySizeClass(p, GetSizeClass(p));
  }
  bool PointerIsMine(void *p) {
    return reinterpret_cast<uptr>(p) / kSpaceSize == kSpaceBeg / kSpaceSize;
  }
  uptr GetSizeClass(void *p) {
    return (reinterpret_cast<uptr>(p) / kRegionSize) % kNumClasses;
  }

  uptr TotalMemoryUsedIncludingFreeLists() {
    uptr res = 0;
    for (uptr i = 0; i < kNumClasses; i++)
      res += GetRegionInfo(i)->allocated;
    return res;
  }

  // Test-only.
  void TestOnlyUnmap() {
    UnmapOrDie(reinterpret_cast<void*>(AllocBeg()), AllocSize());
  }
 private:
  static const uptr kNumClasses = 256;  // Power of two <= 256
  COMPILER_CHECK(kNumClasses <= SizeClassMap::kNumClasses);
  static const uptr kRegionSize = kSpaceSize / kNumClasses;
  // Populate the free list with at most this number of bytes at once
  // or with one element if its size is greater.
  static const uptr kPopulateSize = 1 << 18;

  struct LifoListNode {
    LifoListNode *next;
  };

  struct RegionInfo {
    uptr mutex;  // FIXME
    LifoListNode *free_list;
    uptr allocated;
    char padding[kCacheLineSize -
                 sizeof(mutex) - sizeof(free_list) - sizeof(allocated)];
  };
  COMPILER_CHECK(sizeof(RegionInfo) == kCacheLineSize);

  uptr AdditionalSize() { return sizeof(RegionInfo) * kNumClasses; }
  uptr AllocBeg()  { return kSpaceBeg  - AdditionalSize(); }
  uptr AllocSize() { return kSpaceSize + AdditionalSize(); }

  RegionInfo *GetRegionInfo(uptr class_id) {
    CHECK_LT(class_id, kNumClasses);
    RegionInfo *regions = reinterpret_cast<RegionInfo*>(kSpaceBeg);
    return &regions[-1 - class_id];
  }

  void PushLifoList(LifoListNode **list, LifoListNode *node) {
    node->next = *list;
    *list = node;
  }

  LifoListNode *PopLifoList(LifoListNode **list) {
    LifoListNode *res = *list;
    *list = (*list)->next;
    return res;
  }

  LifoListNode *PopulateFreeList(uptr class_id, RegionInfo *region) {
    uptr size = SizeClassMap::Size(class_id);
    uptr beg_idx = region->allocated;
    uptr end_idx = beg_idx + kPopulateSize;
    LifoListNode *res = 0;
    uptr region_beg = kSpaceBeg + kRegionSize * class_id;
    uptr idx = beg_idx;
    do {  // do-while loop because we need to put at least one item.
      uptr p = region_beg + idx;
      PushLifoList(&res, reinterpret_cast<LifoListNode*>(p));
      idx += size;
    } while (idx < end_idx);
    CHECK_LT(idx, kRegionSize);
    region->allocated += idx - beg_idx;
    return res;
  }

  void *AllocateBySizeClass(uptr class_id) {
    CHECK_LT(class_id, kNumClasses);
    RegionInfo *region = GetRegionInfo(class_id);
    // FIXME: Lock region->mutex;
    if (!region->free_list) {
      region->free_list = PopulateFreeList(class_id, region);
    }
    CHECK_NE(region->free_list, 0);
    LifoListNode *node = PopLifoList(&region->free_list);
    return reinterpret_cast<void*>(node);
  }

  void DeallocateBySizeClass(void *p, uptr class_id) {
    RegionInfo *region = GetRegionInfo(class_id);
    // FIXME: Lock region->mutex;
    PushLifoList(&region->free_list, reinterpret_cast<LifoListNode*>(p));
  }
};

}  // namespace __sanitizer

#endif  // SANITIZER_ALLOCATOR_H
