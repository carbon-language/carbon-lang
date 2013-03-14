//===-- sanitizer_allocator.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Specialized memory allocator for ThreadSanitizer, MemorySanitizer, etc.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_ALLOCATOR_H
#define SANITIZER_ALLOCATOR_H

#include "sanitizer_internal_defs.h"
#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_list.h"
#include "sanitizer_mutex.h"
#include "sanitizer_lfstack.h"

namespace __sanitizer {

// SizeClassMap maps allocation sizes into size classes and back.
// Class 0 corresponds to size 0.
// Classes 1 - 16 correspond to sizes 16 to 256 (size = class_id * 16).
// Next 4 classes: 256 + i * 64  (i = 1 to 4).
// Next 4 classes: 512 + i * 128 (i = 1 to 4).
// ...
// Next 4 classes: 2^k + i * 2^(k-2) (i = 1 to 4).
// Last class corresponds to kMaxSize = 1 << kMaxSizeLog.
//
// This structure of the size class map gives us:
//   - Efficient table-free class-to-size and size-to-class functions.
//   - Difference between two consequent size classes is betweed 14% and 25%
//
// This class also gives a hint to a thread-caching allocator about the amount
// of chunks that need to be cached per-thread:
//  - kMaxNumCached is the maximal number of chunks per size class.
//  - (1 << kMaxBytesCachedLog) is the maximal number of bytes per size class.
//
// Part of output of SizeClassMap::Print():
// c00 => s: 0 diff: +0 00% l 0 cached: 0 0; id 0
// c01 => s: 16 diff: +16 00% l 4 cached: 256 4096; id 1
// c02 => s: 32 diff: +16 100% l 5 cached: 256 8192; id 2
// c03 => s: 48 diff: +16 50% l 5 cached: 256 12288; id 3
// c04 => s: 64 diff: +16 33% l 6 cached: 256 16384; id 4
// c05 => s: 80 diff: +16 25% l 6 cached: 256 20480; id 5
// c06 => s: 96 diff: +16 20% l 6 cached: 256 24576; id 6
// c07 => s: 112 diff: +16 16% l 6 cached: 256 28672; id 7
//
// c08 => s: 128 diff: +16 14% l 7 cached: 256 32768; id 8
// c09 => s: 144 diff: +16 12% l 7 cached: 256 36864; id 9
// c10 => s: 160 diff: +16 11% l 7 cached: 256 40960; id 10
// c11 => s: 176 diff: +16 10% l 7 cached: 256 45056; id 11
// c12 => s: 192 diff: +16 09% l 7 cached: 256 49152; id 12
// c13 => s: 208 diff: +16 08% l 7 cached: 256 53248; id 13
// c14 => s: 224 diff: +16 07% l 7 cached: 256 57344; id 14
// c15 => s: 240 diff: +16 07% l 7 cached: 256 61440; id 15
//
// c16 => s: 256 diff: +16 06% l 8 cached: 256 65536; id 16
// c17 => s: 320 diff: +64 25% l 8 cached: 204 65280; id 17
// c18 => s: 384 diff: +64 20% l 8 cached: 170 65280; id 18
// c19 => s: 448 diff: +64 16% l 8 cached: 146 65408; id 19
//
// c20 => s: 512 diff: +64 14% l 9 cached: 128 65536; id 20
// c21 => s: 640 diff: +128 25% l 9 cached: 102 65280; id 21
// c22 => s: 768 diff: +128 20% l 9 cached: 85 65280; id 22
// c23 => s: 896 diff: +128 16% l 9 cached: 73 65408; id 23
//
// c24 => s: 1024 diff: +128 14% l 10 cached: 64 65536; id 24
// c25 => s: 1280 diff: +256 25% l 10 cached: 51 65280; id 25
// c26 => s: 1536 diff: +256 20% l 10 cached: 42 64512; id 26
// c27 => s: 1792 diff: +256 16% l 10 cached: 36 64512; id 27
//
// ...
//
// c48 => s: 65536 diff: +8192 14% l 16 cached: 1 65536; id 48
// c49 => s: 81920 diff: +16384 25% l 16 cached: 1 81920; id 49
// c50 => s: 98304 diff: +16384 20% l 16 cached: 1 98304; id 50
// c51 => s: 114688 diff: +16384 16% l 16 cached: 1 114688; id 51
//
// c52 => s: 131072 diff: +16384 14% l 17 cached: 1 131072; id 52

template <uptr kMaxSizeLog, uptr kMaxNumCachedT, uptr kMaxBytesCachedLog>
class SizeClassMap {
  static const uptr kMinSizeLog = 4;
  static const uptr kMidSizeLog = kMinSizeLog + 4;
  static const uptr kMinSize = 1 << kMinSizeLog;
  static const uptr kMidSize = 1 << kMidSizeLog;
  static const uptr kMidClass = kMidSize / kMinSize;
  static const uptr S = 2;
  static const uptr M = (1 << S) - 1;

 public:
  static const uptr kMaxNumCached = kMaxNumCachedT;
  // We transfer chunks between central and thread-local free lists in batches.
  // For small size classes we allocate batches separately.
  // For large size classes we use one of the chunks to store the batch.
  struct TransferBatch {
    TransferBatch *next;
    uptr count;
    void *batch[kMaxNumCached];
  };

  static const uptr kMaxSize = 1 << kMaxSizeLog;
  static const uptr kNumClasses =
      kMidClass + ((kMaxSizeLog - kMidSizeLog) << S) + 1;
  COMPILER_CHECK(kNumClasses >= 32 && kNumClasses <= 256);
  static const uptr kNumClassesRounded =
      kNumClasses == 32  ? 32 :
      kNumClasses <= 64  ? 64 :
      kNumClasses <= 128 ? 128 : 256;

  static uptr Size(uptr class_id) {
    if (class_id <= kMidClass)
      return kMinSize * class_id;
    class_id -= kMidClass;
    uptr t = kMidSize << (class_id >> S);
    return t + (t >> S) * (class_id & M);
  }

  static uptr ClassID(uptr size) {
    if (size <= kMidSize)
      return (size + kMinSize - 1) >> kMinSizeLog;
    if (size > kMaxSize) return 0;
    uptr l = MostSignificantSetBitIndex(size);
    uptr hbits = (size >> (l - S)) & M;
    uptr lbits = size & ((1 << (l - S)) - 1);
    uptr l1 = l - kMidSizeLog;
    return kMidClass + (l1 << S) + hbits + (lbits > 0);
  }

  static uptr MaxCached(uptr class_id) {
    if (class_id == 0) return 0;
    uptr n = (1UL << kMaxBytesCachedLog) / Size(class_id);
    return Max<uptr>(1, Min(kMaxNumCached, n));
  }

  static void Print() {
    uptr prev_s = 0;
    uptr total_cached = 0;
    for (uptr i = 0; i < kNumClasses; i++) {
      uptr s = Size(i);
      if (s >= kMidSize / 2 && (s & (s - 1)) == 0)
        Printf("\n");
      uptr d = s - prev_s;
      uptr p = prev_s ? (d * 100 / prev_s) : 0;
      uptr l = s ? MostSignificantSetBitIndex(s) : 0;
      uptr cached = MaxCached(i) * s;
      Printf("c%02zd => s: %zd diff: +%zd %02zd%% l %zd "
             "cached: %zd %zd; id %zd\n",
             i, Size(i), d, p, l, MaxCached(i), cached, ClassID(s));
      total_cached += cached;
      prev_s = s;
    }
    Printf("Total cached: %zd\n", total_cached);
  }

  static bool SizeClassRequiresSeparateTransferBatch(uptr class_id) {
    return Size(class_id) < sizeof(TransferBatch) -
        sizeof(uptr) * (kMaxNumCached - MaxCached(class_id));
  }

  static void Validate() {
    for (uptr c = 1; c < kNumClasses; c++) {
      // Printf("Validate: c%zd\n", c);
      uptr s = Size(c);
      CHECK_NE(s, 0U);
      CHECK_EQ(ClassID(s), c);
      if (c != kNumClasses - 1)
        CHECK_EQ(ClassID(s + 1), c + 1);
      CHECK_EQ(ClassID(s - 1), c);
      if (c)
        CHECK_GT(Size(c), Size(c-1));
    }
    CHECK_EQ(ClassID(kMaxSize + 1), 0);

    for (uptr s = 1; s <= kMaxSize; s++) {
      uptr c = ClassID(s);
      // Printf("s%zd => c%zd\n", s, c);
      CHECK_LT(c, kNumClasses);
      CHECK_GE(Size(c), s);
      if (c > 0)
        CHECK_LT(Size(c-1), s);
    }
  }
};

typedef SizeClassMap<17, 256, 16> DefaultSizeClassMap;
typedef SizeClassMap<17, 64,  14> CompactSizeClassMap;
template<class SizeClassAllocator> struct SizeClassAllocatorLocalCache;

// Memory allocator statistics
enum AllocatorStat {
  AllocatorStatMalloced,
  AllocatorStatFreed,
  AllocatorStatMmapped,
  AllocatorStatUnmapped,
  AllocatorStatCount
};

typedef u64 AllocatorStatCounters[AllocatorStatCount];

// Per-thread stats, live in per-thread cache.
class AllocatorStats {
 public:
  void Init() {
    internal_memset(this, 0, sizeof(*this));
  }

  void Add(AllocatorStat i, u64 v) {
    v += atomic_load(&stats_[i], memory_order_relaxed);
    atomic_store(&stats_[i], v, memory_order_relaxed);
  }

  void Set(AllocatorStat i, u64 v) {
    atomic_store(&stats_[i], v, memory_order_relaxed);
  }

  u64 Get(AllocatorStat i) const {
    return atomic_load(&stats_[i], memory_order_relaxed);
  }

 private:
  friend class AllocatorGlobalStats;
  AllocatorStats *next_;
  AllocatorStats *prev_;
  atomic_uint64_t stats_[AllocatorStatCount];
};

// Global stats, used for aggregation and querying.
class AllocatorGlobalStats : public AllocatorStats {
 public:
  void Init() {
    internal_memset(this, 0, sizeof(*this));
    next_ = this;
    prev_ = this;
  }

  void Register(AllocatorStats *s) {
    SpinMutexLock l(&mu_);
    s->next_ = next_;
    s->prev_ = this;
    next_->prev_ = s;
    next_ = s;
  }

  void Unregister(AllocatorStats *s) {
    SpinMutexLock l(&mu_);
    s->prev_->next_ = s->next_;
    s->next_->prev_ = s->prev_;
    for (int i = 0; i < AllocatorStatCount; i++)
      Add(AllocatorStat(i), s->Get(AllocatorStat(i)));
  }

  void Get(AllocatorStatCounters s) const {
    internal_memset(s, 0, AllocatorStatCount * sizeof(u64));
    SpinMutexLock l(&mu_);
    const AllocatorStats *stats = this;
    for (;;) {
      for (int i = 0; i < AllocatorStatCount; i++)
        s[i] += stats->Get(AllocatorStat(i));
      stats = stats->next_;
      if (stats == this)
        break;
    }
  }

 private:
  mutable SpinMutex mu_;
};

// Allocators call these callbacks on mmap/munmap.
struct NoOpMapUnmapCallback {
  void OnMap(uptr p, uptr size) const { }
  void OnUnmap(uptr p, uptr size) const { }
};

// SizeClassAllocator64 -- allocator for 64-bit address space.
//
// Space: a portion of address space of kSpaceSize bytes starting at
// a fixed address (kSpaceBeg). Both constants are powers of two and
// kSpaceBeg is kSpaceSize-aligned.
// At the beginning the entire space is mprotect-ed, then small parts of it
// are mapped on demand.
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
          const uptr kMetadataSize, class SizeClassMap,
          class MapUnmapCallback = NoOpMapUnmapCallback>
class SizeClassAllocator64 {
 public:
  typedef typename SizeClassMap::TransferBatch Batch;
  typedef SizeClassAllocator64<kSpaceBeg, kSpaceSize, kMetadataSize,
      SizeClassMap, MapUnmapCallback> ThisT;
  typedef SizeClassAllocatorLocalCache<ThisT> AllocatorCache;

  void Init() {
    CHECK_EQ(kSpaceBeg,
             reinterpret_cast<uptr>(Mprotect(kSpaceBeg, kSpaceSize)));
    MapWithCallback(kSpaceEnd, AdditionalSize());
  }

  void MapWithCallback(uptr beg, uptr size) {
    CHECK_EQ(beg, reinterpret_cast<uptr>(MmapFixedOrDie(beg, size)));
    MapUnmapCallback().OnMap(beg, size);
  }

  void UnmapWithCallback(uptr beg, uptr size) {
    MapUnmapCallback().OnUnmap(beg, size);
    UnmapOrDie(reinterpret_cast<void *>(beg), size);
  }

  static bool CanAllocate(uptr size, uptr alignment) {
    return size <= SizeClassMap::kMaxSize &&
      alignment <= SizeClassMap::kMaxSize;
  }

  NOINLINE Batch* AllocateBatch(AllocatorStats *stat, AllocatorCache *c,
                                uptr class_id) {
    CHECK_LT(class_id, kNumClasses);
    RegionInfo *region = GetRegionInfo(class_id);
    Batch *b = region->free_list.Pop();
    if (b == 0)
      b = PopulateFreeList(stat, c, class_id, region);
    region->n_allocated += b->count;
    return b;
  }

  NOINLINE void DeallocateBatch(AllocatorStats *stat, uptr class_id, Batch *b) {
    RegionInfo *region = GetRegionInfo(class_id);
    CHECK_GT(b->count, 0);
    region->free_list.Push(b);
    region->n_freed += b->count;
  }

  static bool PointerIsMine(void *p) {
    return reinterpret_cast<uptr>(p) / kSpaceSize == kSpaceBeg / kSpaceSize;
  }

  static uptr GetSizeClass(void *p) {
    return (reinterpret_cast<uptr>(p) / kRegionSize) % kNumClassesRounded;
  }

  void *GetBlockBegin(void *p) {
    uptr class_id = GetSizeClass(p);
    uptr size = SizeClassMap::Size(class_id);
    if (!size) return 0;
    uptr chunk_idx = GetChunkIdx((uptr)p, size);
    uptr reg_beg = (uptr)p & ~(kRegionSize - 1);
    uptr beg = chunk_idx * size;
    uptr next_beg = beg + size;
    if (class_id >= kNumClasses) return 0;
    RegionInfo *region = GetRegionInfo(class_id);
    if (region->mapped_user >= next_beg)
      return reinterpret_cast<void*>(reg_beg + beg);
    return 0;
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
    UnmapWithCallback(kSpaceBeg, kSpaceSize + AdditionalSize());
  }

  void PrintStats() {
    uptr total_mapped = 0;
    uptr n_allocated = 0;
    uptr n_freed = 0;
    for (uptr class_id = 1; class_id < kNumClasses; class_id++) {
      RegionInfo *region = GetRegionInfo(class_id);
      total_mapped += region->mapped_user;
      n_allocated += region->n_allocated;
      n_freed += region->n_freed;
    }
    Printf("Stats: SizeClassAllocator64: %zdM mapped in %zd allocations; "
           "remains %zd\n",
           total_mapped >> 20, n_allocated, n_allocated - n_freed);
    for (uptr class_id = 1; class_id < kNumClasses; class_id++) {
      RegionInfo *region = GetRegionInfo(class_id);
      if (region->mapped_user == 0) continue;
      Printf("  %02zd (%zd): total: %zd K allocs: %zd remains: %zd\n",
             class_id,
             SizeClassMap::Size(class_id),
             region->mapped_user >> 10,
             region->n_allocated,
             region->n_allocated - region->n_freed);
    }
  }

  // ForceLock() and ForceUnlock() are needed to implement Darwin malloc zone
  // introspection API.
  void ForceLock() {
    for (uptr i = 0; i < kNumClasses; i++) {
      GetRegionInfo(i)->mutex.Lock();
    }
  }

  void ForceUnlock() {
    for (int i = (int)kNumClasses - 1; i >= 0; i--) {
      GetRegionInfo(i)->mutex.Unlock();
    }
  }

  typedef SizeClassMap SizeClassMapT;
  static const uptr kNumClasses = SizeClassMap::kNumClasses;
  static const uptr kNumClassesRounded = SizeClassMap::kNumClassesRounded;

 private:
  static const uptr kRegionSize = kSpaceSize / kNumClassesRounded;
  static const uptr kSpaceEnd = kSpaceBeg + kSpaceSize;
  COMPILER_CHECK(kSpaceBeg % kSpaceSize == 0);
  // kRegionSize must be >= 2^32.
  COMPILER_CHECK((kRegionSize) >= (1ULL << (SANITIZER_WORDSIZE / 2)));
  // Populate the free list with at most this number of bytes at once
  // or with one element if its size is greater.
  static const uptr kPopulateSize = 1 << 14;
  // Call mmap for user memory with at least this size.
  static const uptr kUserMapSize = 1 << 16;
  // Call mmap for metadata memory with at least this size.
  static const uptr kMetaMapSize = 1 << 16;

  struct RegionInfo {
    BlockingMutex mutex;
    LFStack<Batch> free_list;
    uptr allocated_user;  // Bytes allocated for user memory.
    uptr allocated_meta;  // Bytes allocated for metadata.
    uptr mapped_user;  // Bytes mapped for user memory.
    uptr mapped_meta;  // Bytes mapped for metadata.
    uptr n_allocated, n_freed;  // Just stats.
  };
  COMPILER_CHECK(sizeof(RegionInfo) >= kCacheLineSize);

  static uptr AdditionalSize() {
    return RoundUpTo(sizeof(RegionInfo) * kNumClassesRounded,
                     GetPageSizeCached());
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

  NOINLINE Batch* PopulateFreeList(AllocatorStats *stat, AllocatorCache *c,
                                   uptr class_id, RegionInfo *region) {
    BlockingMutexLock l(&region->mutex);
    Batch *b = region->free_list.Pop();
    if (b)
      return b;
    uptr size = SizeClassMap::Size(class_id);
    uptr count = size < kPopulateSize ? SizeClassMap::MaxCached(class_id) : 1;
    uptr beg_idx = region->allocated_user;
    uptr end_idx = beg_idx + count * size;
    uptr region_beg = kSpaceBeg + kRegionSize * class_id;
    if (end_idx + size > region->mapped_user) {
      // Do the mmap for the user memory.
      uptr map_size = kUserMapSize;
      while (end_idx + size > region->mapped_user + map_size)
        map_size += kUserMapSize;
      CHECK_GE(region->mapped_user + map_size, end_idx);
      MapWithCallback(region_beg + region->mapped_user, map_size);
      stat->Add(AllocatorStatMmapped, map_size);
      region->mapped_user += map_size;
    }
    uptr total_count = (region->mapped_user - beg_idx - size)
        / size / count * count;
    region->allocated_meta += total_count * kMetadataSize;
    if (region->allocated_meta > region->mapped_meta) {
      uptr map_size = kMetaMapSize;
      while (region->allocated_meta > region->mapped_meta + map_size)
        map_size += kMetaMapSize;
      // Do the mmap for the metadata.
      CHECK_GE(region->mapped_meta + map_size, region->allocated_meta);
      MapWithCallback(region_beg + kRegionSize -
                      region->mapped_meta - map_size, map_size);
      region->mapped_meta += map_size;
    }
    CHECK_LE(region->allocated_meta, region->mapped_meta);
    if (region->allocated_user + region->allocated_meta > kRegionSize) {
      Printf("%s: Out of memory. Dying. ", SanitizerToolName);
      Printf("The process has exhausted %zuMB for size class %zu.\n",
          kRegionSize / 1024 / 1024, size);
      Die();
    }
    for (;;) {
      if (SizeClassMap::SizeClassRequiresSeparateTransferBatch(class_id))
        b = (Batch*)c->Allocate(this, SizeClassMap::ClassID(sizeof(Batch)));
      else
        b = (Batch*)(region_beg + beg_idx);
      b->count = count;
      for (uptr i = 0; i < count; i++)
        b->batch[i] = (void*)(region_beg + beg_idx + i * size);
      region->allocated_user += count * size;
      CHECK_LE(region->allocated_user, region->mapped_user);
      beg_idx += count * size;
      if (beg_idx + count * size + size > region->mapped_user)
        break;
      CHECK_GT(b->count, 0);
      region->free_list.Push(b);
    }
    return b;
  }
};

// SizeClassAllocator32 -- allocator for 32-bit address space.
// This allocator can theoretically be used on 64-bit arch, but there it is less
// efficient than SizeClassAllocator64.
//
// [kSpaceBeg, kSpaceBeg + kSpaceSize) is the range of addresses which can
// be returned by MmapOrDie().
//
// Region:
//   a result of a single call to MmapAlignedOrDie(kRegionSize, kRegionSize).
// Since the regions are aligned by kRegionSize, there are exactly
// kNumPossibleRegions possible regions in the address space and so we keep
// an u8 array possible_regions[kNumPossibleRegions] to store the size classes.
// 0 size class means the region is not used by the allocator.
//
// One Region is used to allocate chunks of a single size class.
// A Region looks like this:
// UserChunk1 .. UserChunkN <gap> MetaChunkN .. MetaChunk1
//
// In order to avoid false sharing the objects of this class should be
// chache-line aligned.
template <const uptr kSpaceBeg, const u64 kSpaceSize,
          const uptr kMetadataSize, class SizeClassMap,
          class MapUnmapCallback = NoOpMapUnmapCallback>
class SizeClassAllocator32 {
 public:
  typedef typename SizeClassMap::TransferBatch Batch;
  typedef SizeClassAllocator32<kSpaceBeg, kSpaceSize, kMetadataSize,
      SizeClassMap, MapUnmapCallback> ThisT;
  typedef SizeClassAllocatorLocalCache<ThisT> AllocatorCache;

  void Init() {
    state_ = reinterpret_cast<State *>(MapWithCallback(sizeof(State)));
  }

  void *MapWithCallback(uptr size) {
    size = RoundUpTo(size, GetPageSizeCached());
    void *res = MmapOrDie(size, "SizeClassAllocator32");
    MapUnmapCallback().OnMap((uptr)res, size);
    return res;
  }

  void UnmapWithCallback(uptr beg, uptr size) {
    MapUnmapCallback().OnUnmap(beg, size);
    UnmapOrDie(reinterpret_cast<void *>(beg), size);
  }

  static bool CanAllocate(uptr size, uptr alignment) {
    return size <= SizeClassMap::kMaxSize &&
      alignment <= SizeClassMap::kMaxSize;
  }

  void *GetMetaData(void *p) {
    CHECK(PointerIsMine(p));
    uptr mem = reinterpret_cast<uptr>(p);
    uptr beg = ComputeRegionBeg(mem);
    uptr size = SizeClassMap::Size(GetSizeClass(p));
    u32 offset = mem - beg;
    uptr n = offset / (u32)size;  // 32-bit division
    uptr meta = (beg + kRegionSize) - (n + 1) * kMetadataSize;
    return reinterpret_cast<void*>(meta);
  }

  NOINLINE Batch* AllocateBatch(AllocatorStats *stat, AllocatorCache *c,
                                uptr class_id) {
    CHECK_LT(class_id, kNumClasses);
    SizeClassInfo *sci = GetSizeClassInfo(class_id);
    SpinMutexLock l(&sci->mutex);
    if (sci->free_list.empty())
      PopulateFreeList(stat, c, sci, class_id);
    CHECK(!sci->free_list.empty());
    Batch *b = sci->free_list.front();
    sci->free_list.pop_front();
    return b;
  }

  NOINLINE void DeallocateBatch(AllocatorStats *stat, uptr class_id, Batch *b) {
    CHECK_LT(class_id, kNumClasses);
    SizeClassInfo *sci = GetSizeClassInfo(class_id);
    SpinMutexLock l(&sci->mutex);
    CHECK_GT(b->count, 0);
    sci->free_list.push_front(b);
  }

  bool PointerIsMine(void *p) {
    return GetSizeClass(p) != 0;
  }

  uptr GetSizeClass(void *p) {
    return state_->possible_regions[ComputeRegionId(reinterpret_cast<uptr>(p))];
  }

  void *GetBlockBegin(void *p) {
    CHECK(PointerIsMine(p));
    uptr mem = reinterpret_cast<uptr>(p);
    uptr beg = ComputeRegionBeg(mem);
    uptr size = SizeClassMap::Size(GetSizeClass(p));
    u32 offset = mem - beg;
    u32 n = offset / (u32)size;  // 32-bit division
    uptr res = beg + (n * (u32)size);
    return reinterpret_cast<void*>(res);
  }

  uptr GetActuallyAllocatedSize(void *p) {
    CHECK(PointerIsMine(p));
    return SizeClassMap::Size(GetSizeClass(p));
  }

  uptr ClassID(uptr size) { return SizeClassMap::ClassID(size); }

  uptr TotalMemoryUsed() {
    // No need to lock here.
    uptr res = 0;
    for (uptr i = 0; i < kNumPossibleRegions; i++)
      if (state_->possible_regions[i])
        res += kRegionSize;
    return res;
  }

  void TestOnlyUnmap() {
    for (uptr i = 0; i < kNumPossibleRegions; i++)
      if (state_->possible_regions[i])
        UnmapWithCallback((i * kRegionSize), kRegionSize);
    UnmapWithCallback(reinterpret_cast<uptr>(state_), sizeof(State));
  }

  // ForceLock() and ForceUnlock() are needed to implement Darwin malloc zone
  // introspection API.
  void ForceLock() {
    for (uptr i = 0; i < kNumClasses; i++) {
      GetSizeClassInfo(i)->mutex.Lock();
    }
  }

  void ForceUnlock() {
    for (int i = kNumClasses - 1; i >= 0; i--) {
      GetSizeClassInfo(i)->mutex.Unlock();
    }
  }

  void PrintStats() {
  }

  typedef SizeClassMap SizeClassMapT;
  static const uptr kNumClasses = SizeClassMap::kNumClasses;

 private:
  static const uptr kRegionSizeLog = SANITIZER_WORDSIZE == 64 ? 24 : 20;
  static const uptr kRegionSize = 1 << kRegionSizeLog;
  static const uptr kNumPossibleRegions = kSpaceSize / kRegionSize;

  struct SizeClassInfo {
    SpinMutex mutex;
    IntrusiveList<Batch> free_list;
    char padding[kCacheLineSize - sizeof(uptr) - sizeof(IntrusiveList<Batch>)];
  };
  COMPILER_CHECK(sizeof(SizeClassInfo) == kCacheLineSize);

  uptr ComputeRegionId(uptr mem) {
    uptr res = mem >> kRegionSizeLog;
    CHECK_LT(res, kNumPossibleRegions);
    return res;
  }

  uptr ComputeRegionBeg(uptr mem) {
    return mem & ~(kRegionSize - 1);
  }

  uptr AllocateRegion(AllocatorStats *stat, uptr class_id) {
    CHECK_LT(class_id, kNumClasses);
    uptr res = reinterpret_cast<uptr>(MmapAlignedOrDie(kRegionSize, kRegionSize,
                                      "SizeClassAllocator32"));
    MapUnmapCallback().OnMap(res, kRegionSize);
    stat->Add(AllocatorStatMmapped, kRegionSize);
    CHECK_EQ(0U, (res & (kRegionSize - 1)));
    CHECK_EQ(0U, state_->possible_regions[ComputeRegionId(res)]);
    state_->possible_regions[ComputeRegionId(res)] = class_id;
    return res;
  }

  SizeClassInfo *GetSizeClassInfo(uptr class_id) {
    CHECK_LT(class_id, kNumClasses);
    return &state_->size_class_info_array[class_id];
  }

  void PopulateFreeList(AllocatorStats *stat, AllocatorCache *c,
                        SizeClassInfo *sci, uptr class_id) {
    uptr size = SizeClassMap::Size(class_id);
    uptr reg = AllocateRegion(stat, class_id);
    uptr n_chunks = kRegionSize / (size + kMetadataSize);
    uptr max_count = SizeClassMap::MaxCached(class_id);
    Batch *b = 0;
    for (uptr i = reg; i < reg + n_chunks * size; i += size) {
      if (b == 0) {
        if (SizeClassMap::SizeClassRequiresSeparateTransferBatch(class_id))
          b = (Batch*)c->Allocate(this, SizeClassMap::ClassID(sizeof(Batch)));
        else
          b = (Batch*)i;
        b->count = 0;
      }
      b->batch[b->count++] = (void*)i;
      if (b->count == max_count) {
        CHECK_GT(b->count, 0);
        sci->free_list.push_back(b);
        b = 0;
      }
    }
    if (b) {
      CHECK_GT(b->count, 0);
      sci->free_list.push_back(b);
    }
  }

  struct State {
    u8 possible_regions[kNumPossibleRegions];
    SizeClassInfo size_class_info_array[kNumClasses];
  };
  State *state_;
};

// Objects of this type should be used as local caches for SizeClassAllocator64
// or SizeClassAllocator32. Since the typical use of this class is to have one
// object per thread in TLS, is has to be POD.
template<class SizeClassAllocator>
struct SizeClassAllocatorLocalCache {
  typedef SizeClassAllocator Allocator;
  static const uptr kNumClasses = SizeClassAllocator::kNumClasses;

  void Init(AllocatorGlobalStats *s) {
    stats_.Init();
    if (s)
      s->Register(&stats_);
  }

  void Destroy(SizeClassAllocator *allocator, AllocatorGlobalStats *s) {
    Drain(allocator);
    if (s)
      s->Unregister(&stats_);
  }

  void *Allocate(SizeClassAllocator *allocator, uptr class_id) {
    CHECK_NE(class_id, 0UL);
    CHECK_LT(class_id, kNumClasses);
    stats_.Add(AllocatorStatMalloced, SizeClassMap::Size(class_id));
    PerClass *c = &per_class_[class_id];
    if (UNLIKELY(c->count == 0))
      Refill(allocator, class_id);
    void *res = c->batch[--c->count];
    PREFETCH(c->batch[c->count - 1]);
    return res;
  }

  void Deallocate(SizeClassAllocator *allocator, uptr class_id, void *p) {
    CHECK_NE(class_id, 0UL);
    CHECK_LT(class_id, kNumClasses);
    // If the first allocator call on a new thread is a deallocation, then
    // max_count will be zero, leading to check failure.
    InitCache();
    stats_.Add(AllocatorStatFreed, SizeClassMap::Size(class_id));
    PerClass *c = &per_class_[class_id];
    CHECK_NE(c->max_count, 0UL);
    if (UNLIKELY(c->count == c->max_count))
      Drain(allocator, class_id);
    c->batch[c->count++] = p;
  }

  void Drain(SizeClassAllocator *allocator) {
    for (uptr class_id = 0; class_id < kNumClasses; class_id++) {
      PerClass *c = &per_class_[class_id];
      while (c->count > 0)
        Drain(allocator, class_id);
    }
  }

  // private:
  typedef typename SizeClassAllocator::SizeClassMapT SizeClassMap;
  typedef typename SizeClassMap::TransferBatch Batch;
  struct PerClass {
    uptr count;
    uptr max_count;
    void *batch[2 * SizeClassMap::kMaxNumCached];
  };
  PerClass per_class_[kNumClasses];
  AllocatorStats stats_;

  void InitCache() {
    if (per_class_[1].max_count)
      return;
    for (uptr i = 0; i < kNumClasses; i++) {
      PerClass *c = &per_class_[i];
      c->max_count = 2 * SizeClassMap::MaxCached(i);
    }
  }

  NOINLINE void Refill(SizeClassAllocator *allocator, uptr class_id) {
    InitCache();
    PerClass *c = &per_class_[class_id];
    Batch *b = allocator->AllocateBatch(&stats_, this, class_id);
    CHECK_GT(b->count, 0);
    for (uptr i = 0; i < b->count; i++)
      c->batch[i] = b->batch[i];
    c->count = b->count;
    if (SizeClassMap::SizeClassRequiresSeparateTransferBatch(class_id))
      Deallocate(allocator, SizeClassMap::ClassID(sizeof(Batch)), b);
  }

  NOINLINE void Drain(SizeClassAllocator *allocator, uptr class_id) {
    InitCache();
    PerClass *c = &per_class_[class_id];
    Batch *b;
    if (SizeClassMap::SizeClassRequiresSeparateTransferBatch(class_id))
      b = (Batch*)Allocate(allocator, SizeClassMap::ClassID(sizeof(Batch)));
    else
      b = (Batch*)c->batch[0];
    uptr cnt = Min(c->max_count / 2, c->count);
    for (uptr i = 0; i < cnt; i++) {
      b->batch[i] = c->batch[i];
      c->batch[i] = c->batch[i + c->max_count / 2];
    }
    b->count = cnt;
    c->count -= cnt;
    CHECK_GT(b->count, 0);
    allocator->DeallocateBatch(&stats_, class_id, b);
  }
};

// This class can (de)allocate only large chunks of memory using mmap/unmap.
// The main purpose of this allocator is to cover large and rare allocation
// sizes not covered by more efficient allocators (e.g. SizeClassAllocator64).
template <class MapUnmapCallback = NoOpMapUnmapCallback>
class LargeMmapAllocator {
 public:
  void Init() {
    internal_memset(this, 0, sizeof(*this));
    page_size_ = GetPageSizeCached();
  }

  void *Allocate(AllocatorStats *stat, uptr size, uptr alignment) {
    CHECK(IsPowerOfTwo(alignment));
    uptr map_size = RoundUpMapSize(size);
    if (alignment > page_size_)
      map_size += alignment;
    if (map_size < size) return 0;  // Overflow.
    uptr map_beg = reinterpret_cast<uptr>(
        MmapOrDie(map_size, "LargeMmapAllocator"));
    MapUnmapCallback().OnMap(map_beg, map_size);
    uptr map_end = map_beg + map_size;
    uptr res = map_beg + page_size_;
    if (res & (alignment - 1))  // Align.
      res += alignment - (res & (alignment - 1));
    CHECK_EQ(0, res & (alignment - 1));
    CHECK_LE(res + size, map_end);
    Header *h = GetHeader(res);
    h->size = size;
    h->map_beg = map_beg;
    h->map_size = map_size;
    uptr size_log = MostSignificantSetBitIndex(map_size);
    CHECK_LT(size_log, ARRAY_SIZE(stats.by_size_log));
    {
      SpinMutexLock l(&mutex_);
      uptr idx = n_chunks_++;
      CHECK_LT(idx, kMaxNumChunks);
      h->chunk_idx = idx;
      chunks_[idx] = h;
      stats.n_allocs++;
      stats.currently_allocated += map_size;
      stats.max_allocated = Max(stats.max_allocated, stats.currently_allocated);
      stats.by_size_log[size_log]++;
      stat->Add(AllocatorStatMalloced, map_size);
      stat->Add(AllocatorStatMmapped, map_size);
    }
    return reinterpret_cast<void*>(res);
  }

  void Deallocate(AllocatorStats *stat, void *p) {
    Header *h = GetHeader(p);
    {
      SpinMutexLock l(&mutex_);
      uptr idx = h->chunk_idx;
      CHECK_EQ(chunks_[idx], h);
      CHECK_LT(idx, n_chunks_);
      chunks_[idx] = chunks_[n_chunks_ - 1];
      chunks_[idx]->chunk_idx = idx;
      n_chunks_--;
      stats.n_frees++;
      stats.currently_allocated -= h->map_size;
      stat->Add(AllocatorStatFreed, h->map_size);
      stat->Add(AllocatorStatUnmapped, h->map_size);
    }
    MapUnmapCallback().OnUnmap(h->map_beg, h->map_size);
    UnmapOrDie(reinterpret_cast<void*>(h->map_beg), h->map_size);
  }

  uptr TotalMemoryUsed() {
    SpinMutexLock l(&mutex_);
    uptr res = 0;
    for (uptr i = 0; i < n_chunks_; i++) {
      Header *h = chunks_[i];
      CHECK_EQ(h->chunk_idx, i);
      res += RoundUpMapSize(h->size);
    }
    return res;
  }

  bool PointerIsMine(void *p) {
    return GetBlockBegin(p) != 0;
  }

  uptr GetActuallyAllocatedSize(void *p) {
    return RoundUpTo(GetHeader(p)->size, page_size_);
  }

  // At least page_size_/2 metadata bytes is available.
  void *GetMetaData(void *p) {
    // Too slow: CHECK_EQ(p, GetBlockBegin(p));
    CHECK(IsAligned(reinterpret_cast<uptr>(p), page_size_));
    return GetHeader(p) + 1;
  }

  void *GetBlockBegin(void *ptr) {
    uptr p = reinterpret_cast<uptr>(ptr);
    SpinMutexLock l(&mutex_);
    uptr nearest_chunk = 0;
    // Cache-friendly linear search.
    for (uptr i = 0; i < n_chunks_; i++) {
      uptr ch = reinterpret_cast<uptr>(chunks_[i]);
      if (p < ch) continue;  // p is at left to this chunk, skip it.
      if (p - ch < p - nearest_chunk)
        nearest_chunk = ch;
    }
    if (!nearest_chunk)
      return 0;
    Header *h = reinterpret_cast<Header *>(nearest_chunk);
    CHECK_GE(nearest_chunk, h->map_beg);
    CHECK_LT(nearest_chunk, h->map_beg + h->map_size);
    CHECK_LE(nearest_chunk, p);
    if (h->map_beg + h->map_size < p)
      return 0;
    return GetUser(h);
  }

  void PrintStats() {
    Printf("Stats: LargeMmapAllocator: allocated %zd times, "
           "remains %zd (%zd K) max %zd M; by size logs: ",
           stats.n_allocs, stats.n_allocs - stats.n_frees,
           stats.currently_allocated >> 10, stats.max_allocated >> 20);
    for (uptr i = 0; i < ARRAY_SIZE(stats.by_size_log); i++) {
      uptr c = stats.by_size_log[i];
      if (!c) continue;
      Printf("%zd:%zd; ", i, c);
    }
    Printf("\n");
  }

  // ForceLock() and ForceUnlock() are needed to implement Darwin malloc zone
  // introspection API.
  void ForceLock() {
    mutex_.Lock();
  }

  void ForceUnlock() {
    mutex_.Unlock();
  }

 private:
  static const int kMaxNumChunks = 1 << FIRST_32_SECOND_64(15, 18);
  struct Header {
    uptr map_beg;
    uptr map_size;
    uptr size;
    uptr chunk_idx;
  };

  Header *GetHeader(uptr p) {
    CHECK_EQ(p % page_size_, 0);
    return reinterpret_cast<Header*>(p - page_size_);
  }
  Header *GetHeader(void *p) { return GetHeader(reinterpret_cast<uptr>(p)); }

  void *GetUser(Header *h) {
    CHECK_EQ((uptr)h % page_size_, 0);
    return reinterpret_cast<void*>(reinterpret_cast<uptr>(h) + page_size_);
  }

  uptr RoundUpMapSize(uptr size) {
    return RoundUpTo(size, page_size_) + page_size_;
  }

  uptr page_size_;
  Header *chunks_[kMaxNumChunks];
  uptr n_chunks_;
  struct Stats {
    uptr n_allocs, n_frees, currently_allocated, max_allocated, by_size_log[64];
  } stats;
  SpinMutex mutex_;
};

// This class implements a complete memory allocator by using two
// internal allocators:
// PrimaryAllocator is efficient, but may not allocate some sizes (alignments).
//  When allocating 2^x bytes it should return 2^x aligned chunk.
// PrimaryAllocator is used via a local AllocatorCache.
// SecondaryAllocator can allocate anything, but is not efficient.
template <class PrimaryAllocator, class AllocatorCache,
          class SecondaryAllocator>  // NOLINT
class CombinedAllocator {
 public:
  void Init() {
    primary_.Init();
    secondary_.Init();
    stats_.Init();
  }

  void *Allocate(AllocatorCache *cache, uptr size, uptr alignment,
                 bool cleared = false) {
    // Returning 0 on malloc(0) may break a lot of code.
    if (size == 0)
      size = 1;
    if (size + alignment < size)
      return 0;
    if (alignment > 8)
      size = RoundUpTo(size, alignment);
    void *res;
    if (primary_.CanAllocate(size, alignment))
      res = cache->Allocate(&primary_, primary_.ClassID(size));
    else
      res = secondary_.Allocate(&stats_, size, alignment);
    if (alignment > 8)
      CHECK_EQ(reinterpret_cast<uptr>(res) & (alignment - 1), 0);
    if (cleared && res)
      internal_memset(res, 0, size);
    return res;
  }

  void Deallocate(AllocatorCache *cache, void *p) {
    if (!p) return;
    if (primary_.PointerIsMine(p))
      cache->Deallocate(&primary_, primary_.GetSizeClass(p), p);
    else
      secondary_.Deallocate(&stats_, p);
  }

  void *Reallocate(AllocatorCache *cache, void *p, uptr new_size,
                   uptr alignment) {
    if (!p)
      return Allocate(cache, new_size, alignment);
    if (!new_size) {
      Deallocate(cache, p);
      return 0;
    }
    CHECK(PointerIsMine(p));
    uptr old_size = GetActuallyAllocatedSize(p);
    uptr memcpy_size = Min(new_size, old_size);
    void *new_p = Allocate(cache, new_size, alignment);
    if (new_p)
      internal_memcpy(new_p, p, memcpy_size);
    Deallocate(cache, p);
    return new_p;
  }

  bool PointerIsMine(void *p) {
    if (primary_.PointerIsMine(p))
      return true;
    return secondary_.PointerIsMine(p);
  }

  bool FromPrimary(void *p) {
    return primary_.PointerIsMine(p);
  }

  void *GetMetaData(void *p) {
    if (primary_.PointerIsMine(p))
      return primary_.GetMetaData(p);
    return secondary_.GetMetaData(p);
  }

  void *GetBlockBegin(void *p) {
    if (primary_.PointerIsMine(p))
      return primary_.GetBlockBegin(p);
    return secondary_.GetBlockBegin(p);
  }

  uptr GetActuallyAllocatedSize(void *p) {
    if (primary_.PointerIsMine(p))
      return primary_.GetActuallyAllocatedSize(p);
    return secondary_.GetActuallyAllocatedSize(p);
  }

  uptr TotalMemoryUsed() {
    return primary_.TotalMemoryUsed() + secondary_.TotalMemoryUsed();
  }

  void TestOnlyUnmap() { primary_.TestOnlyUnmap(); }

  void InitCache(AllocatorCache *cache) {
    cache->Init(&stats_);
  }

  void DestroyCache(AllocatorCache *cache) {
    cache->Destroy(&primary_, &stats_);
  }

  void SwallowCache(AllocatorCache *cache) {
    cache->Drain(&primary_);
  }

  void GetStats(AllocatorStatCounters s) const {
    stats_.Get(s);
  }

  void PrintStats() {
    primary_.PrintStats();
    secondary_.PrintStats();
  }

  // ForceLock() and ForceUnlock() are needed to implement Darwin malloc zone
  // introspection API.
  void ForceLock() {
    primary_.ForceLock();
    secondary_.ForceLock();
  }

  void ForceUnlock() {
    secondary_.ForceUnlock();
    primary_.ForceUnlock();
  }

 private:
  PrimaryAllocator primary_;
  SecondaryAllocator secondary_;
  AllocatorGlobalStats stats_;
};

// Returns true if calloc(size, n) should return 0 due to overflow in size*n.
bool CallocShouldReturnNullDueToOverflow(uptr size, uptr n);

}  // namespace __sanitizer

#endif  // SANITIZER_ALLOCATOR_H

