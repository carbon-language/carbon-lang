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
//===----------------------------------------------------------------------===//
#include "asan_allocator.h"

#include "asan_mapping.h"
#include "asan_poisoning.h"
#include "asan_report.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_list.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_quarantine.h"
#include "lsan/lsan_common.h"

namespace __asan {

struct AsanMapUnmapCallback {
  void OnMap(uptr p, uptr size) const {
    PoisonShadow(p, size, kAsanHeapLeftRedzoneMagic);
    // Statistics.
    AsanStats &thread_stats = GetCurrentThreadStats();
    thread_stats.mmaps++;
    thread_stats.mmaped += size;
  }
  void OnUnmap(uptr p, uptr size) const {
    PoisonShadow(p, size, 0);
    // We are about to unmap a chunk of user memory.
    // Mark the corresponding shadow memory as not needed.
    // Since asan's mapping is compacting, the shadow chunk may be
    // not page-aligned, so we only flush the page-aligned portion.
    uptr page_size = GetPageSizeCached();
    uptr shadow_beg = RoundUpTo(MemToShadow(p), page_size);
    uptr shadow_end = RoundDownTo(MemToShadow(p + size), page_size);
    FlushUnneededShadowMemory(shadow_beg, shadow_end - shadow_beg);
    // Statistics.
    AsanStats &thread_stats = GetCurrentThreadStats();
    thread_stats.munmaps++;
    thread_stats.munmaped += size;
  }
};

#if SANITIZER_WORDSIZE == 64
#if defined(__powerpc64__)
const uptr kAllocatorSpace =  0xa0000000000ULL;
const uptr kAllocatorSize  =  0x20000000000ULL;  // 2T.
#else
const uptr kAllocatorSpace = 0x600000000000ULL;
const uptr kAllocatorSize  =  0x40000000000ULL;  // 4T.
#endif
typedef DefaultSizeClassMap SizeClassMap;
typedef SizeClassAllocator64<kAllocatorSpace, kAllocatorSize, 0 /*metadata*/,
    SizeClassMap, AsanMapUnmapCallback> PrimaryAllocator;
#elif SANITIZER_WORDSIZE == 32
static const u64 kAddressSpaceSize = 1ULL << 32;
typedef CompactSizeClassMap SizeClassMap;
static const uptr kRegionSizeLog = 20;
static const uptr kFlatByteMapSize = kAddressSpaceSize >> kRegionSizeLog;
typedef SizeClassAllocator32<0, kAddressSpaceSize, 16,
  SizeClassMap, kRegionSizeLog,
  FlatByteMap<kFlatByteMapSize>,
  AsanMapUnmapCallback> PrimaryAllocator;
#endif

typedef SizeClassAllocatorLocalCache<PrimaryAllocator> AllocatorCache;
typedef LargeMmapAllocator<AsanMapUnmapCallback> SecondaryAllocator;
typedef CombinedAllocator<PrimaryAllocator, AllocatorCache,
    SecondaryAllocator> Allocator;

// We can not use THREADLOCAL because it is not supported on some of the
// platforms we care about (OSX 10.6, Android).
// static THREADLOCAL AllocatorCache cache;
AllocatorCache *GetAllocatorCache(AsanThreadLocalMallocStorage *ms) {
  CHECK(ms);
  CHECK_LE(sizeof(AllocatorCache), sizeof(ms->allocator2_cache));
  return reinterpret_cast<AllocatorCache *>(ms->allocator2_cache);
}

static Allocator allocator;

static const uptr kMaxAllowedMallocSize =
  FIRST_32_SECOND_64(3UL << 30, 8UL << 30);

static const uptr kMaxThreadLocalQuarantine =
  FIRST_32_SECOND_64(1 << 18, 1 << 20);

// Every chunk of memory allocated by this allocator can be in one of 3 states:
// CHUNK_AVAILABLE: the chunk is in the free list and ready to be allocated.
// CHUNK_ALLOCATED: the chunk is allocated and not yet freed.
// CHUNK_QUARANTINE: the chunk was freed and put into quarantine zone.
enum {
  CHUNK_AVAILABLE  = 0,  // 0 is the default value even if we didn't set it.
  CHUNK_ALLOCATED  = 2,
  CHUNK_QUARANTINE = 3
};

// Valid redzone sizes are 16, 32, 64, ... 2048, so we encode them in 3 bits.
// We use adaptive redzones: for larger allocation larger redzones are used.
static u32 RZLog2Size(u32 rz_log) {
  CHECK_LT(rz_log, 8);
  return 16 << rz_log;
}

static u32 RZSize2Log(u32 rz_size) {
  CHECK_GE(rz_size, 16);
  CHECK_LE(rz_size, 2048);
  CHECK(IsPowerOfTwo(rz_size));
  u32 res = Log2(rz_size) - 4;
  CHECK_EQ(rz_size, RZLog2Size(res));
  return res;
}

static uptr ComputeRZLog(uptr user_requested_size) {
  u32 rz_log =
    user_requested_size <= 64        - 16   ? 0 :
    user_requested_size <= 128       - 32   ? 1 :
    user_requested_size <= 512       - 64   ? 2 :
    user_requested_size <= 4096      - 128  ? 3 :
    user_requested_size <= (1 << 14) - 256  ? 4 :
    user_requested_size <= (1 << 15) - 512  ? 5 :
    user_requested_size <= (1 << 16) - 1024 ? 6 : 7;
  return Max(rz_log, RZSize2Log(flags()->redzone));
}

// The memory chunk allocated from the underlying allocator looks like this:
// L L L L L L H H U U U U U U R R
//   L -- left redzone words (0 or more bytes)
//   H -- ChunkHeader (16 bytes), which is also a part of the left redzone.
//   U -- user memory.
//   R -- right redzone (0 or more bytes)
// ChunkBase consists of ChunkHeader and other bytes that overlap with user
// memory.

// If the left redzone is greater than the ChunkHeader size we store a magic
// value in the first uptr word of the memory block and store the address of
// ChunkBase in the next uptr.
// M B L L L L L L L L L  H H U U U U U U
//   |                    ^
//   ---------------------|
//   M -- magic value kAllocBegMagic
//   B -- address of ChunkHeader pointing to the first 'H'
static const uptr kAllocBegMagic = 0xCC6E96B9;

struct ChunkHeader {
  // 1-st 8 bytes.
  u32 chunk_state       : 8;  // Must be first.
  u32 alloc_tid         : 24;

  u32 free_tid          : 24;
  u32 from_memalign     : 1;
  u32 alloc_type        : 2;
  u32 rz_log            : 3;
  u32 lsan_tag          : 2;
  // 2-nd 8 bytes
  // This field is used for small sizes. For large sizes it is equal to
  // SizeClassMap::kMaxSize and the actual size is stored in the
  // SecondaryAllocator's metadata.
  u32 user_requested_size;
  u32 alloc_context_id;
};

struct ChunkBase : ChunkHeader {
  // Header2, intersects with user memory.
  u32 free_context_id;
};

static const uptr kChunkHeaderSize = sizeof(ChunkHeader);
static const uptr kChunkHeader2Size = sizeof(ChunkBase) - kChunkHeaderSize;
COMPILER_CHECK(kChunkHeaderSize == 16);
COMPILER_CHECK(kChunkHeader2Size <= 16);

struct AsanChunk: ChunkBase {
  uptr Beg() { return reinterpret_cast<uptr>(this) + kChunkHeaderSize; }
  uptr UsedSize() {
    if (user_requested_size != SizeClassMap::kMaxSize)
      return user_requested_size;
    return *reinterpret_cast<uptr *>(allocator.GetMetaData(AllocBeg()));
  }
  void *AllocBeg() {
    if (from_memalign)
      return allocator.GetBlockBegin(reinterpret_cast<void *>(this));
    return reinterpret_cast<void*>(Beg() - RZLog2Size(rz_log));
  }
  // If we don't use stack depot, we store the alloc/free stack traces
  // in the chunk itself.
  u32 *AllocStackBeg() {
    return (u32*)(Beg() - RZLog2Size(rz_log));
  }
  uptr AllocStackSize() {
    CHECK_LE(RZLog2Size(rz_log), kChunkHeaderSize);
    return (RZLog2Size(rz_log) - kChunkHeaderSize) / sizeof(u32);
  }
  u32 *FreeStackBeg() {
    return (u32*)(Beg() + kChunkHeader2Size);
  }
  uptr FreeStackSize() {
    if (user_requested_size < kChunkHeader2Size) return 0;
    uptr available = RoundUpTo(user_requested_size, SHADOW_GRANULARITY);
    return (available - kChunkHeader2Size) / sizeof(u32);
  }
  bool AddrIsInside(uptr addr) {
    return (addr >= Beg()) && (addr < Beg() + UsedSize());
  }
};

uptr AsanChunkView::Beg() { return chunk_->Beg(); }
uptr AsanChunkView::End() { return Beg() + UsedSize(); }
uptr AsanChunkView::UsedSize() { return chunk_->UsedSize(); }
uptr AsanChunkView::AllocTid() { return chunk_->alloc_tid; }
uptr AsanChunkView::FreeTid() { return chunk_->free_tid; }

static void GetStackTraceFromId(u32 id, StackTrace *stack) {
  CHECK(id);
  uptr size = 0;
  const uptr *trace = StackDepotGet(id, &size);
  CHECK_LT(size, kStackTraceMax);
  internal_memcpy(stack->trace, trace, sizeof(uptr) * size);
  stack->size = size;
}

void AsanChunkView::GetAllocStack(StackTrace *stack) {
  if (flags()->use_stack_depot)
    GetStackTraceFromId(chunk_->alloc_context_id, stack);
  else
    StackTrace::UncompressStack(stack, chunk_->AllocStackBeg(),
                                chunk_->AllocStackSize());
}

void AsanChunkView::GetFreeStack(StackTrace *stack) {
  if (flags()->use_stack_depot)
    GetStackTraceFromId(chunk_->free_context_id, stack);
  else
    StackTrace::UncompressStack(stack, chunk_->FreeStackBeg(),
                                chunk_->FreeStackSize());
}

struct QuarantineCallback;
typedef Quarantine<QuarantineCallback, AsanChunk> AsanQuarantine;
typedef AsanQuarantine::Cache QuarantineCache;
static AsanQuarantine quarantine(LINKER_INITIALIZED);
static QuarantineCache fallback_quarantine_cache(LINKER_INITIALIZED);
static AllocatorCache fallback_allocator_cache;
static SpinMutex fallback_mutex;

QuarantineCache *GetQuarantineCache(AsanThreadLocalMallocStorage *ms) {
  CHECK(ms);
  CHECK_LE(sizeof(QuarantineCache), sizeof(ms->quarantine_cache));
  return reinterpret_cast<QuarantineCache *>(ms->quarantine_cache);
}

struct QuarantineCallback {
  explicit QuarantineCallback(AllocatorCache *cache)
      : cache_(cache) {
  }

  void Recycle(AsanChunk *m) {
    CHECK_EQ(m->chunk_state, CHUNK_QUARANTINE);
    atomic_store((atomic_uint8_t*)m, CHUNK_AVAILABLE, memory_order_relaxed);
    CHECK_NE(m->alloc_tid, kInvalidTid);
    CHECK_NE(m->free_tid, kInvalidTid);
    PoisonShadow(m->Beg(),
                 RoundUpTo(m->UsedSize(), SHADOW_GRANULARITY),
                 kAsanHeapLeftRedzoneMagic);
    void *p = reinterpret_cast<void *>(m->AllocBeg());
    if (p != m) {
      uptr *alloc_magic = reinterpret_cast<uptr *>(p);
      CHECK_EQ(alloc_magic[0], kAllocBegMagic);
      // Clear the magic value, as allocator internals may overwrite the
      // contents of deallocated chunk, confusing GetAsanChunk lookup.
      alloc_magic[0] = 0;
      CHECK_EQ(alloc_magic[1], reinterpret_cast<uptr>(m));
    }

    // Statistics.
    AsanStats &thread_stats = GetCurrentThreadStats();
    thread_stats.real_frees++;
    thread_stats.really_freed += m->UsedSize();

    allocator.Deallocate(cache_, p);
  }

  void *Allocate(uptr size) {
    return allocator.Allocate(cache_, size, 1, false);
  }

  void Deallocate(void *p) {
    allocator.Deallocate(cache_, p);
  }

  AllocatorCache *cache_;
};

void InitializeAllocator() {
  allocator.Init();
  quarantine.Init((uptr)flags()->quarantine_size, kMaxThreadLocalQuarantine);
}

static void *Allocate(uptr size, uptr alignment, StackTrace *stack,
                      AllocType alloc_type, bool can_fill) {
  if (!asan_inited)
    __asan_init();
  Flags &fl = *flags();
  CHECK(stack);
  const uptr min_alignment = SHADOW_GRANULARITY;
  if (alignment < min_alignment)
    alignment = min_alignment;
  if (size == 0) {
    // We'd be happy to avoid allocating memory for zero-size requests, but
    // some programs/tests depend on this behavior and assume that malloc would
    // not return NULL even for zero-size allocations. Moreover, it looks like
    // operator new should never return NULL, and results of consecutive "new"
    // calls must be different even if the allocated size is zero.
    size = 1;
  }
  CHECK(IsPowerOfTwo(alignment));
  uptr rz_log = ComputeRZLog(size);
  uptr rz_size = RZLog2Size(rz_log);
  uptr rounded_size = RoundUpTo(Max(size, kChunkHeader2Size), alignment);
  uptr needed_size = rounded_size + rz_size;
  if (alignment > min_alignment)
    needed_size += alignment;
  bool using_primary_allocator = true;
  // If we are allocating from the secondary allocator, there will be no
  // automatic right redzone, so add the right redzone manually.
  if (!PrimaryAllocator::CanAllocate(needed_size, alignment)) {
    needed_size += rz_size;
    using_primary_allocator = false;
  }
  CHECK(IsAligned(needed_size, min_alignment));
  if (size > kMaxAllowedMallocSize || needed_size > kMaxAllowedMallocSize) {
    Report("WARNING: AddressSanitizer failed to allocate %p bytes\n",
           (void*)size);
    return 0;
  }

  AsanThread *t = GetCurrentThread();
  void *allocated;
  if (t) {
    AllocatorCache *cache = GetAllocatorCache(&t->malloc_storage());
    allocated = allocator.Allocate(cache, needed_size, 8, false);
  } else {
    SpinMutexLock l(&fallback_mutex);
    AllocatorCache *cache = &fallback_allocator_cache;
    allocated = allocator.Allocate(cache, needed_size, 8, false);
  }
  uptr alloc_beg = reinterpret_cast<uptr>(allocated);
  uptr alloc_end = alloc_beg + needed_size;
  uptr beg_plus_redzone = alloc_beg + rz_size;
  uptr user_beg = beg_plus_redzone;
  if (!IsAligned(user_beg, alignment))
    user_beg = RoundUpTo(user_beg, alignment);
  uptr user_end = user_beg + size;
  CHECK_LE(user_end, alloc_end);
  uptr chunk_beg = user_beg - kChunkHeaderSize;
  AsanChunk *m = reinterpret_cast<AsanChunk *>(chunk_beg);
  m->alloc_type = alloc_type;
  m->rz_log = rz_log;
  u32 alloc_tid = t ? t->tid() : 0;
  m->alloc_tid = alloc_tid;
  CHECK_EQ(alloc_tid, m->alloc_tid);  // Does alloc_tid fit into the bitfield?
  m->free_tid = kInvalidTid;
  m->from_memalign = user_beg != beg_plus_redzone;
  if (alloc_beg != chunk_beg) {
    CHECK_LE(alloc_beg+ 2 * sizeof(uptr), chunk_beg);
    reinterpret_cast<uptr *>(alloc_beg)[0] = kAllocBegMagic;
    reinterpret_cast<uptr *>(alloc_beg)[1] = chunk_beg;
  }
  if (using_primary_allocator) {
    CHECK(size);
    m->user_requested_size = size;
    CHECK(allocator.FromPrimary(allocated));
  } else {
    CHECK(!allocator.FromPrimary(allocated));
    m->user_requested_size = SizeClassMap::kMaxSize;
    uptr *meta = reinterpret_cast<uptr *>(allocator.GetMetaData(allocated));
    meta[0] = size;
    meta[1] = chunk_beg;
  }

  if (fl.use_stack_depot) {
    m->alloc_context_id = StackDepotPut(stack->trace, stack->size);
  } else {
    m->alloc_context_id = 0;
    StackTrace::CompressStack(stack, m->AllocStackBeg(), m->AllocStackSize());
  }

  uptr size_rounded_down_to_granularity = RoundDownTo(size, SHADOW_GRANULARITY);
  // Unpoison the bulk of the memory region.
  if (size_rounded_down_to_granularity)
    PoisonShadow(user_beg, size_rounded_down_to_granularity, 0);
  // Deal with the end of the region if size is not aligned to granularity.
  if (size != size_rounded_down_to_granularity && fl.poison_heap) {
    u8 *shadow = (u8*)MemToShadow(user_beg + size_rounded_down_to_granularity);
    *shadow = size & (SHADOW_GRANULARITY - 1);
  }

  AsanStats &thread_stats = GetCurrentThreadStats();
  thread_stats.mallocs++;
  thread_stats.malloced += size;
  thread_stats.malloced_redzones += needed_size - size;
  uptr class_id = Min(kNumberOfSizeClasses, SizeClassMap::ClassID(needed_size));
  thread_stats.malloced_by_size[class_id]++;
  if (needed_size > SizeClassMap::kMaxSize)
    thread_stats.malloc_large++;

  void *res = reinterpret_cast<void *>(user_beg);
  if (can_fill && fl.max_malloc_fill_size) {
    uptr fill_size = Min(size, (uptr)fl.max_malloc_fill_size);
    REAL(memset)(res, fl.malloc_fill_byte, fill_size);
  }
  m->lsan_tag = __lsan::DisabledInThisThread() ? __lsan::kIgnored
                                               : __lsan::kDirectlyLeaked;
  // Must be the last mutation of metadata in this function.
  atomic_store((atomic_uint8_t *)m, CHUNK_ALLOCATED, memory_order_release);
  ASAN_MALLOC_HOOK(res, size);
  return res;
}

static void ReportInvalidFree(void *ptr, u8 chunk_state, StackTrace *stack) {
  if (chunk_state == CHUNK_QUARANTINE)
    ReportDoubleFree((uptr)ptr, stack);
  else
    ReportFreeNotMalloced((uptr)ptr, stack);
}

static void AtomicallySetQuarantineFlag(AsanChunk *m,
                                        void *ptr, StackTrace *stack) {
  u8 old_chunk_state = CHUNK_ALLOCATED;
  // Flip the chunk_state atomically to avoid race on double-free.
  if (!atomic_compare_exchange_strong((atomic_uint8_t*)m, &old_chunk_state,
                                      CHUNK_QUARANTINE, memory_order_acquire))
    ReportInvalidFree(ptr, old_chunk_state, stack);
  CHECK_EQ(CHUNK_ALLOCATED, old_chunk_state);
}

// Expects the chunk to already be marked as quarantined by using
// AtomicallySetQuarantineFlag.
static void QuarantineChunk(AsanChunk *m, void *ptr,
                            StackTrace *stack, AllocType alloc_type) {
  CHECK_EQ(m->chunk_state, CHUNK_QUARANTINE);

  if (m->alloc_type != alloc_type && flags()->alloc_dealloc_mismatch)
    ReportAllocTypeMismatch((uptr)ptr, stack,
                            (AllocType)m->alloc_type, (AllocType)alloc_type);

  CHECK_GE(m->alloc_tid, 0);
  if (SANITIZER_WORDSIZE == 64)  // On 32-bits this resides in user area.
    CHECK_EQ(m->free_tid, kInvalidTid);
  AsanThread *t = GetCurrentThread();
  m->free_tid = t ? t->tid() : 0;
  if (flags()->use_stack_depot) {
    m->free_context_id = StackDepotPut(stack->trace, stack->size);
  } else {
    m->free_context_id = 0;
    StackTrace::CompressStack(stack, m->FreeStackBeg(), m->FreeStackSize());
  }
  // Poison the region.
  PoisonShadow(m->Beg(),
               RoundUpTo(m->UsedSize(), SHADOW_GRANULARITY),
               kAsanHeapFreeMagic);

  AsanStats &thread_stats = GetCurrentThreadStats();
  thread_stats.frees++;
  thread_stats.freed += m->UsedSize();

  // Push into quarantine.
  if (t) {
    AsanThreadLocalMallocStorage *ms = &t->malloc_storage();
    AllocatorCache *ac = GetAllocatorCache(ms);
    quarantine.Put(GetQuarantineCache(ms), QuarantineCallback(ac),
                   m, m->UsedSize());
  } else {
    SpinMutexLock l(&fallback_mutex);
    AllocatorCache *ac = &fallback_allocator_cache;
    quarantine.Put(&fallback_quarantine_cache, QuarantineCallback(ac),
                   m, m->UsedSize());
  }
}

static void Deallocate(void *ptr, StackTrace *stack, AllocType alloc_type) {
  uptr p = reinterpret_cast<uptr>(ptr);
  if (p == 0) return;

  uptr chunk_beg = p - kChunkHeaderSize;
  AsanChunk *m = reinterpret_cast<AsanChunk *>(chunk_beg);
  ASAN_FREE_HOOK(ptr);
  // Must mark the chunk as quarantined before any changes to its metadata.
  AtomicallySetQuarantineFlag(m, ptr, stack);
  QuarantineChunk(m, ptr, stack, alloc_type);
}

static void *Reallocate(void *old_ptr, uptr new_size, StackTrace *stack) {
  CHECK(old_ptr && new_size);
  uptr p = reinterpret_cast<uptr>(old_ptr);
  uptr chunk_beg = p - kChunkHeaderSize;
  AsanChunk *m = reinterpret_cast<AsanChunk *>(chunk_beg);

  AsanStats &thread_stats = GetCurrentThreadStats();
  thread_stats.reallocs++;
  thread_stats.realloced += new_size;

  void *new_ptr = Allocate(new_size, 8, stack, FROM_MALLOC, true);
  if (new_ptr) {
    u8 chunk_state = m->chunk_state;
    if (chunk_state != CHUNK_ALLOCATED)
      ReportInvalidFree(old_ptr, chunk_state, stack);
    CHECK_NE(REAL(memcpy), (void*)0);
    uptr memcpy_size = Min(new_size, m->UsedSize());
    // If realloc() races with free(), we may start copying freed memory.
    // However, we will report racy double-free later anyway.
    REAL(memcpy)(new_ptr, old_ptr, memcpy_size);
    Deallocate(old_ptr, stack, FROM_MALLOC);
  }
  return new_ptr;
}

// Assumes alloc_beg == allocator.GetBlockBegin(alloc_beg).
static AsanChunk *GetAsanChunk(void *alloc_beg) {
  if (!alloc_beg) return 0;
  if (!allocator.FromPrimary(alloc_beg)) {
    uptr *meta = reinterpret_cast<uptr *>(allocator.GetMetaData(alloc_beg));
    AsanChunk *m = reinterpret_cast<AsanChunk *>(meta[1]);
    return m;
  }
  uptr *alloc_magic = reinterpret_cast<uptr *>(alloc_beg);
  if (alloc_magic[0] == kAllocBegMagic)
    return reinterpret_cast<AsanChunk *>(alloc_magic[1]);
  return reinterpret_cast<AsanChunk *>(alloc_beg);
}

static AsanChunk *GetAsanChunkByAddr(uptr p) {
  void *alloc_beg = allocator.GetBlockBegin(reinterpret_cast<void *>(p));
  return GetAsanChunk(alloc_beg);
}

// Allocator must be locked when this function is called.
static AsanChunk *GetAsanChunkByAddrFastLocked(uptr p) {
  void *alloc_beg =
      allocator.GetBlockBeginFastLocked(reinterpret_cast<void *>(p));
  return GetAsanChunk(alloc_beg);
}

static uptr AllocationSize(uptr p) {
  AsanChunk *m = GetAsanChunkByAddr(p);
  if (!m) return 0;
  if (m->chunk_state != CHUNK_ALLOCATED) return 0;
  if (m->Beg() != p) return 0;
  return m->UsedSize();
}

// We have an address between two chunks, and we want to report just one.
AsanChunk *ChooseChunk(uptr addr,
                       AsanChunk *left_chunk, AsanChunk *right_chunk) {
  // Prefer an allocated chunk over freed chunk and freed chunk
  // over available chunk.
  if (left_chunk->chunk_state != right_chunk->chunk_state) {
    if (left_chunk->chunk_state == CHUNK_ALLOCATED)
      return left_chunk;
    if (right_chunk->chunk_state == CHUNK_ALLOCATED)
      return right_chunk;
    if (left_chunk->chunk_state == CHUNK_QUARANTINE)
      return left_chunk;
    if (right_chunk->chunk_state == CHUNK_QUARANTINE)
      return right_chunk;
  }
  // Same chunk_state: choose based on offset.
  sptr l_offset = 0, r_offset = 0;
  CHECK(AsanChunkView(left_chunk).AddrIsAtRight(addr, 1, &l_offset));
  CHECK(AsanChunkView(right_chunk).AddrIsAtLeft(addr, 1, &r_offset));
  if (l_offset < r_offset)
    return left_chunk;
  return right_chunk;
}

AsanChunkView FindHeapChunkByAddress(uptr addr) {
  AsanChunk *m1 = GetAsanChunkByAddr(addr);
  if (!m1) return AsanChunkView(m1);
  sptr offset = 0;
  if (AsanChunkView(m1).AddrIsAtLeft(addr, 1, &offset)) {
    // The address is in the chunk's left redzone, so maybe it is actually
    // a right buffer overflow from the other chunk to the left.
    // Search a bit to the left to see if there is another chunk.
    AsanChunk *m2 = 0;
    for (uptr l = 1; l < GetPageSizeCached(); l++) {
      m2 = GetAsanChunkByAddr(addr - l);
      if (m2 == m1) continue;  // Still the same chunk.
      break;
    }
    if (m2 && AsanChunkView(m2).AddrIsAtRight(addr, 1, &offset))
      m1 = ChooseChunk(addr, m2, m1);
  }
  return AsanChunkView(m1);
}

void AsanThreadLocalMallocStorage::CommitBack() {
  AllocatorCache *ac = GetAllocatorCache(this);
  quarantine.Drain(GetQuarantineCache(this), QuarantineCallback(ac));
  allocator.SwallowCache(GetAllocatorCache(this));
}

void PrintInternalAllocatorStats() {
  allocator.PrintStats();
}

SANITIZER_INTERFACE_ATTRIBUTE
void *asan_memalign(uptr alignment, uptr size, StackTrace *stack,
                    AllocType alloc_type) {
  return Allocate(size, alignment, stack, alloc_type, true);
}

SANITIZER_INTERFACE_ATTRIBUTE
void asan_free(void *ptr, StackTrace *stack, AllocType alloc_type) {
  Deallocate(ptr, stack, alloc_type);
}

SANITIZER_INTERFACE_ATTRIBUTE
void *asan_malloc(uptr size, StackTrace *stack) {
  return Allocate(size, 8, stack, FROM_MALLOC, true);
}

void *asan_calloc(uptr nmemb, uptr size, StackTrace *stack) {
  if (CallocShouldReturnNullDueToOverflow(size, nmemb)) return 0;
  void *ptr = Allocate(nmemb * size, 8, stack, FROM_MALLOC, false);
  // If the memory comes from the secondary allocator no need to clear it
  // as it comes directly from mmap.
  if (ptr && allocator.FromPrimary(ptr))
    REAL(memset)(ptr, 0, nmemb * size);
  return ptr;
}

void *asan_realloc(void *p, uptr size, StackTrace *stack) {
  if (p == 0)
    return Allocate(size, 8, stack, FROM_MALLOC, true);
  if (size == 0) {
    Deallocate(p, stack, FROM_MALLOC);
    return 0;
  }
  return Reallocate(p, size, stack);
}

void *asan_valloc(uptr size, StackTrace *stack) {
  return Allocate(size, GetPageSizeCached(), stack, FROM_MALLOC, true);
}

void *asan_pvalloc(uptr size, StackTrace *stack) {
  uptr PageSize = GetPageSizeCached();
  size = RoundUpTo(size, PageSize);
  if (size == 0) {
    // pvalloc(0) should allocate one page.
    size = PageSize;
  }
  return Allocate(size, PageSize, stack, FROM_MALLOC, true);
}

int asan_posix_memalign(void **memptr, uptr alignment, uptr size,
                        StackTrace *stack) {
  void *ptr = Allocate(size, alignment, stack, FROM_MALLOC, true);
  CHECK(IsAligned((uptr)ptr, alignment));
  *memptr = ptr;
  return 0;
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr asan_malloc_usable_size(void *ptr, StackTrace *stack) {
  CHECK(stack);
  if (ptr == 0) return 0;
  uptr usable_size = AllocationSize(reinterpret_cast<uptr>(ptr));
  if (flags()->check_malloc_usable_size && (usable_size == 0))
    ReportMallocUsableSizeNotOwned((uptr)ptr, stack);
  return usable_size;
}

uptr asan_mz_size(const void *ptr) {
  return AllocationSize(reinterpret_cast<uptr>(ptr));
}

void asan_mz_force_lock() {
  allocator.ForceLock();
  fallback_mutex.Lock();
}

void asan_mz_force_unlock() {
  fallback_mutex.Unlock();
  allocator.ForceUnlock();
}

}  // namespace __asan

// --- Implementation of LSan-specific functions --- {{{1
namespace __lsan {
void LockAllocator() {
  __asan::allocator.ForceLock();
}

void UnlockAllocator() {
  __asan::allocator.ForceUnlock();
}

void GetAllocatorGlobalRange(uptr *begin, uptr *end) {
  *begin = (uptr)&__asan::allocator;
  *end = *begin + sizeof(__asan::allocator);
}

void *PointsIntoChunk(void* p) {
  uptr addr = reinterpret_cast<uptr>(p);
  __asan::AsanChunk *m = __asan::GetAsanChunkByAddrFastLocked(addr);
  if (!m) return 0;
  uptr chunk = m->Beg();
  if ((m->chunk_state == __asan::CHUNK_ALLOCATED) && m->AddrIsInside(addr))
    return reinterpret_cast<void *>(chunk);
  return 0;
}

void *GetUserBegin(void *p) {
  __asan::AsanChunk *m =
      __asan::GetAsanChunkByAddrFastLocked(reinterpret_cast<uptr>(p));
  CHECK(m);
  return reinterpret_cast<void *>(m->Beg());
}

LsanMetadata::LsanMetadata(void *chunk) {
  uptr addr = reinterpret_cast<uptr>(chunk);
  metadata_ = reinterpret_cast<void *>(addr - __asan::kChunkHeaderSize);
}

bool LsanMetadata::allocated() const {
  __asan::AsanChunk *m = reinterpret_cast<__asan::AsanChunk *>(metadata_);
  return m->chunk_state == __asan::CHUNK_ALLOCATED;
}

ChunkTag LsanMetadata::tag() const {
  __asan::AsanChunk *m = reinterpret_cast<__asan::AsanChunk *>(metadata_);
  return static_cast<ChunkTag>(m->lsan_tag);
}

void LsanMetadata::set_tag(ChunkTag value) {
  __asan::AsanChunk *m = reinterpret_cast<__asan::AsanChunk *>(metadata_);
  m->lsan_tag = value;
}

uptr LsanMetadata::requested_size() const {
  __asan::AsanChunk *m = reinterpret_cast<__asan::AsanChunk *>(metadata_);
  return m->UsedSize();
}

u32 LsanMetadata::stack_trace_id() const {
  __asan::AsanChunk *m = reinterpret_cast<__asan::AsanChunk *>(metadata_);
  return m->alloc_context_id;
}

template <typename Callable> void ForEachChunk(Callable const &callback) {
  __asan::allocator.ForEachChunk(callback);
}
#if CAN_SANITIZE_LEAKS
template void ForEachChunk<ProcessPlatformSpecificAllocationsCb>(
    ProcessPlatformSpecificAllocationsCb const &callback);
template void ForEachChunk<PrintLeakedCb>(PrintLeakedCb const &callback);
template void ForEachChunk<CollectLeaksCb>(CollectLeaksCb const &callback);
template void ForEachChunk<MarkIndirectlyLeakedCb>(
    MarkIndirectlyLeakedCb const &callback);
template void ForEachChunk<CollectIgnoredCb>(
    CollectIgnoredCb const &callback);
#endif  // CAN_SANITIZE_LEAKS

IgnoreObjectResult IgnoreObjectLocked(const void *p) {
  uptr addr = reinterpret_cast<uptr>(p);
  __asan::AsanChunk *m = __asan::GetAsanChunkByAddr(addr);
  if (!m) return kIgnoreObjectInvalid;
  if ((m->chunk_state == __asan::CHUNK_ALLOCATED) && m->AddrIsInside(addr)) {
    if (m->lsan_tag == kIgnored)
      return kIgnoreObjectAlreadyIgnored;
    m->lsan_tag = __lsan::kIgnored;
    return kIgnoreObjectSuccess;
  } else {
    return kIgnoreObjectInvalid;
  }
}
}  // namespace __lsan

// ---------------------- Interface ---------------- {{{1
using namespace __asan;  // NOLINT

// ASan allocator doesn't reserve extra bytes, so normally we would
// just return "size". We don't want to expose our redzone sizes, etc here.
uptr __asan_get_estimated_allocated_size(uptr size) {
  return size;
}

bool __asan_get_ownership(const void *p) {
  uptr ptr = reinterpret_cast<uptr>(p);
  return (AllocationSize(ptr) > 0);
}

uptr __asan_get_allocated_size(const void *p) {
  if (p == 0) return 0;
  uptr ptr = reinterpret_cast<uptr>(p);
  uptr allocated_size = AllocationSize(ptr);
  // Die if p is not malloced or if it is already freed.
  if (allocated_size == 0) {
    GET_STACK_TRACE_FATAL_HERE;
    ReportAsanGetAllocatedSizeNotOwned(ptr, &stack);
  }
  return allocated_size;
}

#if !SANITIZER_SUPPORTS_WEAK_HOOKS
// Provide default (no-op) implementation of malloc hooks.
extern "C" {
SANITIZER_WEAK_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE
void __asan_malloc_hook(void *ptr, uptr size) {
  (void)ptr;
  (void)size;
}
SANITIZER_WEAK_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE
void __asan_free_hook(void *ptr) {
  (void)ptr;
}
}  // extern "C"
#endif
