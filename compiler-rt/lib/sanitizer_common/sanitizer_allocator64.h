//===-- sanitizer_allocator64.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Specialized allocator which works only in 64-bit address space.
// It is used by ThreadSanitizer, MemorySanitizer and possibly other tools.
// The main feature of this allocator is that the header is located far away
// from the user memory region, so that the tool does not use extra shadow
// for the header.
// Another important feature is that the size class of a pointer is computed
// without any memory accesses by simply looking at the address.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_ALLOCATOR64_H
#define SANITIZER_ALLOCATOR64_H

#include "sanitizer_allocator.h"

#if SANITIZER_WORDSIZE != 64
# error "sanitizer_allocator64.h can only be used on 64-bit platforms"
#endif

namespace __sanitizer {

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

  bool CanAllocate(uptr size, uptr alignment) {
    return size <= SizeClassMap::kMaxSize &&
      alignment <= SizeClassMap::kMaxSize;
  }

  void *Allocate(uptr size, uptr alignment) {
    CHECK(CanAllocate(size, alignment));
    return AllocateBySizeClass(SizeClassMap::ClassID(size));
  }

  void Deallocate(void *p) {
    CHECK(PointerIsMine(p));
    DeallocateBySizeClass(p, GetSizeClass(p));
  }

  // Allocate several chunks of the given class_id.
  void BulkAllocate(uptr class_id, AllocatorFreeList *free_list) {
    CHECK_LT(class_id, kNumClasses);
    RegionInfo *region = GetRegionInfo(class_id);
    SpinMutexLock l(&region->mutex);
    if (region->free_list.empty()) {
      PopulateFreeList(class_id, region);
    }
    CHECK(!region->free_list.empty());
    uptr count = SizeClassMap::MaxCached(class_id);
    if (region->free_list.size() <= count) {
      free_list->append_front(&region->free_list);
    } else {
      for (uptr i = 0; i < count; i++) {
        AllocatorListNode *node = region->free_list.front();
        region->free_list.pop_front();
        free_list->push_front(node);
      }
    }
    CHECK(!free_list->empty());
  }

  // Swallow the entire free_list for the given class_id.
  void BulkDeallocate(uptr class_id, AllocatorFreeList *free_list) {
    CHECK_LT(class_id, kNumClasses);
    RegionInfo *region = GetRegionInfo(class_id);
    SpinMutexLock l(&region->mutex);
    region->free_list.append_front(free_list);
  }

  static bool PointerIsMine(void *p) {
    return reinterpret_cast<uptr>(p) / kSpaceSize == kSpaceBeg / kSpaceSize;
  }

  static uptr GetSizeClass(void *p) {
    return (reinterpret_cast<uptr>(p) / kRegionSize) % kNumClasses;
  }

  static void *GetBlockBegin(void *p) {
    uptr class_id = GetSizeClass(p);
    uptr size = SizeClassMap::Size(class_id);
    uptr chunk_idx = GetChunkIdx((uptr)p, size);
    uptr reg_beg = (uptr)p & ~(kRegionSize - 1);
    uptr begin = reg_beg + chunk_idx * size;
    return (void*)begin;
  }

  static uptr GetActuallyAllocatedSize(void *p) {
    CHECK(PointerIsMine(p));
    return SizeClassMap::Size(GetSizeClass(p));
  }

  uptr ClassID(uptr size) { return SizeClassMap::ClassID(size); }

  void *GetMetaData(void *p) {
    uptr class_id = GetSizeClass(p);
    uptr size = SizeClassMap::Size(class_id);
    uptr chunk_idx = GetChunkIdx(reinterpret_cast<uptr>(p), size);
    return reinterpret_cast<void*>(kSpaceBeg + (kRegionSize * (class_id + 1)) -
                                   (1 + chunk_idx) * kMetadataSize);
  }

  uptr TotalMemoryUsed() {
    uptr res = 0;
    for (uptr i = 0; i < kNumClasses; i++)
      res += GetRegionInfo(i)->allocated_user;
    return res;
  }

  // Test-only.
  void TestOnlyUnmap() {
    UnmapOrDie(reinterpret_cast<void*>(AllocBeg()), AllocSize());
  }

  static uptr AllocBeg()  { return kSpaceBeg; }
  static uptr AllocSize() { return kSpaceSize + AdditionalSize(); }

  typedef SizeClassMap SizeClassMapT;
  static const uptr kNumClasses = SizeClassMap::kNumClasses;  // 2^k <= 256

 private:
  COMPILER_CHECK(kSpaceBeg % kSpaceSize == 0);
  COMPILER_CHECK(kNumClasses <= SizeClassMap::kNumClasses);
  static const uptr kRegionSize = kSpaceSize / kNumClasses;
  COMPILER_CHECK((kRegionSize >> 32) > 0);  // kRegionSize must be >= 2^32.
  // Populate the free list with at most this number of bytes at once
  // or with one element if its size is greater.
  static const uptr kPopulateSize = 1 << 18;

  struct RegionInfo {
    SpinMutex mutex;
    AllocatorFreeList free_list;
    uptr allocated_user;  // Bytes allocated for user memory.
    uptr allocated_meta;  // Bytes allocated for metadata.
    char padding[kCacheLineSize - 3 * sizeof(uptr) - sizeof(AllocatorFreeList)];
  };
  COMPILER_CHECK(sizeof(RegionInfo) == kCacheLineSize);

  static uptr AdditionalSize() {
    uptr res = sizeof(RegionInfo) * kNumClasses;
    CHECK_EQ(res % GetPageSizeCached(), 0);
    return res;
  }

  RegionInfo *GetRegionInfo(uptr class_id) {
    CHECK_LT(class_id, kNumClasses);
    RegionInfo *regions = reinterpret_cast<RegionInfo*>(kSpaceBeg + kSpaceSize);
    return &regions[class_id];
  }

  static uptr GetChunkIdx(uptr chunk, uptr size) {
    u32 offset = chunk % kRegionSize;
    // Here we divide by a non-constant. This is costly.
    // We require that kRegionSize is at least 2^32 so that offset is 32-bit.
    // We save 2x by using 32-bit div, but may need to use a 256-way switch.
    return offset / (u32)size;
  }

  void PopulateFreeList(uptr class_id, RegionInfo *region) {
    uptr size = SizeClassMap::Size(class_id);
    uptr beg_idx = region->allocated_user;
    uptr end_idx = beg_idx + kPopulateSize;
    region->free_list.clear();
    uptr region_beg = kSpaceBeg + kRegionSize * class_id;
    uptr idx = beg_idx;
    uptr i = 0;
    do {  // do-while loop because we need to put at least one item.
      uptr p = region_beg + idx;
      region->free_list.push_front(reinterpret_cast<AllocatorListNode*>(p));
      idx += size;
      i++;
    } while (idx < end_idx);
    region->allocated_user += idx - beg_idx;
    region->allocated_meta += i * kMetadataSize;
    if (region->allocated_user + region->allocated_meta > kRegionSize) {
      Printf("Out of memory. Dying.\n");
      Printf("The process has exhausted %zuMB for size class %zu.\n",
          kRegionSize / 1024 / 1024, size);
      Die();
    }
  }

  void *AllocateBySizeClass(uptr class_id) {
    CHECK_LT(class_id, kNumClasses);
    RegionInfo *region = GetRegionInfo(class_id);
    SpinMutexLock l(&region->mutex);
    if (region->free_list.empty()) {
      PopulateFreeList(class_id, region);
    }
    CHECK(!region->free_list.empty());
    AllocatorListNode *node = region->free_list.front();
    region->free_list.pop_front();
    return reinterpret_cast<void*>(node);
  }

  void DeallocateBySizeClass(void *p, uptr class_id) {
    RegionInfo *region = GetRegionInfo(class_id);
    SpinMutexLock l(&region->mutex);
    region->free_list.push_front(reinterpret_cast<AllocatorListNode*>(p));
  }
};

}  // namespace __sanitizer

#endif  // SANITIZER_ALLOCATOR64_H
