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

#include "asan_mapping.h"
#include "asan_report.h"
#include "asan_thread.h"
#include "asan_thread_registry.h"
#include "sanitizer/asan_interface.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_list.h"

namespace __asan {

struct AsanMapUnmapCallback {
  void OnMap(uptr p, uptr size) const {
    PoisonShadow(p, size, kAsanHeapLeftRedzoneMagic);
    // Statistics.
    AsanStats &thread_stats = asanThreadRegistry().GetCurrentThreadStats();
    thread_stats.mmaps++;
    thread_stats.mmaped += size;
    // thread_stats.mmaped_by_size[size_class] += n_chunks;
  }
  void OnUnmap(uptr p, uptr size) const {
    PoisonShadow(p, size, 0);
    // Statistics.
    AsanStats &thread_stats = asanThreadRegistry().GetCurrentThreadStats();
    thread_stats.munmaps++;
    thread_stats.munmaped += size;
  }
};

#if SANITIZER_WORDSIZE == 64
const uptr kAllocatorSpace = 0x600000000000ULL;
const uptr kAllocatorSize  =  0x10000000000ULL;  // 1T.
typedef DefaultSizeClassMap SizeClassMap;
typedef SizeClassAllocator64<kAllocatorSpace, kAllocatorSize, 0 /*metadata*/,
    SizeClassMap, AsanMapUnmapCallback> PrimaryAllocator;
#elif SANITIZER_WORDSIZE == 32
static const u64 kAddressSpaceSize = 1ULL << 32;
typedef CompactSizeClassMap SizeClassMap;
typedef SizeClassAllocator32<0, kAddressSpaceSize, 16,
  SizeClassMap, AsanMapUnmapCallback> PrimaryAllocator;
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

static const uptr kReturnOnZeroMalloc = 2048;  // Zero page is protected.

static int inited = 0;

static void Init() {
  if (inited) return;
  __asan_init();
  inited = true;  // this must happen before any threads are created.
  allocator.Init();
}

// Every chunk of memory allocated by this allocator can be in one of 3 states:
// CHUNK_AVAILABLE: the chunk is in the free list and ready to be allocated.
// CHUNK_ALLOCATED: the chunk is allocated and not yet freed.
// CHUNK_QUARANTINE: the chunk was freed and put into quarantine zone.
enum {
  CHUNK_AVAILABLE  = 1,
  CHUNK_ALLOCATED  = 2,
  CHUNK_QUARANTINE = 3
};

// The memory chunk allocated from the underlying allocator looks like this:
// L L L L L L H H U U U U U U R R
//   L -- left redzone words (0 or more bytes)
//   H -- ChunkHeader (16 bytes), which is also a part of the left redzone.
//   U -- user memory.
//   R -- right redzone (0 or more bytes)
// ChunkBase consists of ChunkHeader and other bytes that overlap with user
// memory.

// If a memory chunk is allocated by memalign and we had to increase the
// allocation size to achieve the proper alignment, then we store this magic
// value in the first uptr word of the memory block and store the address of
// ChunkBase in the next uptr.
// M B ? ? ? L L L L L L  H H U U U U U U
//   M -- magic value kMemalignMagic
//   B -- address of ChunkHeader pointing to the first 'H'
static const uptr kMemalignMagic = 0xCC6E96B9;

#if SANITIZER_WORDSIZE == 64
struct ChunkBase {
  // 1-st 8 bytes.
  uptr chunk_state       : 8;  // Must be first.
  uptr alloc_tid         : 24;

  uptr free_tid          : 24;
  uptr from_memalign     : 1;
  uptr alloc_type        : 2;
  // 2-nd 8 bytes
  uptr user_requested_size;
  // Header2 (intersects with user memory).
  // 3-rd 8 bytes. These overlap with the user memory.
  AsanChunk *next;
};

static const uptr kChunkHeaderSize = 16;
static const uptr kChunkHeader2Size = 8;

#elif SANITIZER_WORDSIZE == 32
struct ChunkBase {
  // 1-st 8 bytes.
  uptr chunk_state       : 8;  // Must be first.
  uptr alloc_tid         : 24;

  uptr from_memalign     : 1;
  uptr alloc_type        : 2;
  uptr free_tid          : 24;
  // 2-nd 8 bytes
  uptr user_requested_size;
  AsanChunk *next;
  // Header2 empty.
};

static const uptr kChunkHeaderSize = 16;
static const uptr kChunkHeader2Size = 0;
#endif
COMPILER_CHECK(sizeof(ChunkBase) == kChunkHeaderSize + kChunkHeader2Size);

static uptr ComputeRZSize(uptr user_requested_size) {
  // FIXME: implement adaptive redzones.
  return flags()->redzone;
}

struct AsanChunk: ChunkBase {
  uptr Beg() { return reinterpret_cast<uptr>(this) + kChunkHeaderSize; }
  uptr UsedSize() { return user_requested_size; }
  // We store the alloc/free stack traces in the chunk itself.
  u32 *AllocStackBeg() {
    return (u32*)(Beg() - ComputeRZSize(UsedSize()));
  }
  uptr AllocStackSize() {
    return (ComputeRZSize(UsedSize()) - kChunkHeaderSize) / sizeof(u32);
  }
  u32 *FreeStackBeg() {
    return (u32*)(Beg() + kChunkHeader2Size);
  }
  uptr FreeStackSize() {
    uptr available = Max(RoundUpTo(UsedSize(), SHADOW_GRANULARITY),
                         ComputeRZSize(UsedSize()));
    return (available - kChunkHeader2Size) / sizeof(u32);
  }
};

uptr AsanChunkView::Beg() { return chunk_->Beg(); }
uptr AsanChunkView::End() { return Beg() + UsedSize(); }
uptr AsanChunkView::UsedSize() { return chunk_->UsedSize(); }
uptr AsanChunkView::AllocTid() { return chunk_->alloc_tid; }
uptr AsanChunkView::FreeTid() { return chunk_->free_tid; }

void AsanChunkView::GetAllocStack(StackTrace *stack) {
  StackTrace::UncompressStack(stack, chunk_->AllocStackBeg(),
                              chunk_->AllocStackSize());
}

void AsanChunkView::GetFreeStack(StackTrace *stack) {
  StackTrace::UncompressStack(stack, chunk_->FreeStackBeg(),
                              chunk_->FreeStackSize());
}

class Quarantine: public AsanChunkFifoList {
 public:
  void SwallowThreadLocalQuarantine(AsanThreadLocalMallocStorage *ms) {
    AsanChunkFifoList *q = &ms->quarantine_;
    if (!q->size()) return;
    SpinMutexLock l(&mutex_);
    PushList(q);
    PopAndDeallocateLoop(ms);
  }

  void BypassThreadLocalQuarantine(AsanChunk *m) {
    SpinMutexLock l(&mutex_);
    Push(m);
  }

 private:
  void PopAndDeallocateLoop(AsanThreadLocalMallocStorage *ms) {
    while (size() > (uptr)flags()->quarantine_size) {
      PopAndDeallocate(ms);
    }
  }
  void PopAndDeallocate(AsanThreadLocalMallocStorage *ms) {
    CHECK_GT(size(), 0);
    AsanChunk *m = Pop();
    CHECK(m);
    CHECK(m->chunk_state == CHUNK_QUARANTINE);
    m->chunk_state = CHUNK_AVAILABLE;
    CHECK_NE(m->alloc_tid, kInvalidTid);
    CHECK_NE(m->free_tid, kInvalidTid);
    PoisonShadow(m->Beg(),
                 RoundUpTo(m->user_requested_size, SHADOW_GRANULARITY),
                 kAsanHeapLeftRedzoneMagic);
    uptr alloc_beg = m->Beg() - ComputeRZSize(m->user_requested_size);
    void *p = reinterpret_cast<void *>(alloc_beg);
    if (m->from_memalign) {
      p = allocator.GetBlockBegin(p);
      uptr *memalign_magic = reinterpret_cast<uptr *>(p);
      CHECK_EQ(memalign_magic[0], kMemalignMagic);
      CHECK_EQ(memalign_magic[1], reinterpret_cast<uptr>(m));
    }

    // Statistics.
    AsanStats &thread_stats = asanThreadRegistry().GetCurrentThreadStats();
    thread_stats.real_frees++;
    thread_stats.really_freed += m->UsedSize();

    allocator.Deallocate(GetAllocatorCache(ms), p);
  }
  SpinMutex mutex_;
};

static Quarantine quarantine;

void AsanChunkFifoList::PushList(AsanChunkFifoList *q) {
  CHECK(q->size() > 0);
  size_ += q->size();
  append_back(q);
  q->clear();
}

void AsanChunkFifoList::Push(AsanChunk *n) {
  push_back(n);
  size_ += n->UsedSize();
}

// Interesting performance observation: this function takes up to 15% of overal
// allocator time. That's because *first_ has been evicted from cache long time
// ago. Not sure if we can or want to do anything with this.
AsanChunk *AsanChunkFifoList::Pop() {
  CHECK(first_);
  AsanChunk *res = front();
  size_ -= res->UsedSize();
  pop_front();
  return res;
}

static void *Allocate(uptr size, uptr alignment, StackTrace *stack,
                      AllocType alloc_type) {
  Init();
  CHECK(stack);
  if (alignment < 8) alignment = 8;
  if (size == 0) {
    if (alignment <= kReturnOnZeroMalloc)
      return reinterpret_cast<void *>(kReturnOnZeroMalloc);
    else
      return 0;  // 0 bytes with large alignment requested. Just return 0.
  }
  CHECK(IsPowerOfTwo(alignment));
  uptr rz_size = ComputeRZSize(size);
  uptr rounded_size = RoundUpTo(size, rz_size);
  uptr needed_size = rounded_size + rz_size;
  if (alignment > rz_size)
    needed_size += alignment;
  // If we are allocating from the secondary allocator, there will be no
  // automatic right redzone, so add the right redzone manually.
  if (!PrimaryAllocator::CanAllocate(needed_size, alignment))
    needed_size += rz_size;
  CHECK(IsAligned(needed_size, rz_size));
  if (size > kMaxAllowedMallocSize || needed_size > kMaxAllowedMallocSize) {
    Report("WARNING: AddressSanitizer failed to allocate %p bytes\n",
           (void*)size);
    return 0;
  }

  AsanThread *t = asanThreadRegistry().GetCurrent();
  AllocatorCache *cache = t ? GetAllocatorCache(&t->malloc_storage()) : 0;
  void *allocated = allocator.Allocate(cache, needed_size, 8, false);
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
  m->chunk_state = CHUNK_ALLOCATED;
  m->alloc_type = alloc_type;
  u32 alloc_tid = t ? t->tid() : 0;
  m->alloc_tid = alloc_tid;
  CHECK_EQ(alloc_tid, m->alloc_tid);  // Does alloc_tid fit into the bitfield?
  m->free_tid = kInvalidTid;
  m->from_memalign = user_beg != beg_plus_redzone;
  if (m->from_memalign) {
    CHECK_LE(beg_plus_redzone + 2 * sizeof(uptr), user_beg);
    uptr *memalign_magic = reinterpret_cast<uptr *>(alloc_beg);
    memalign_magic[0] = kMemalignMagic;
    memalign_magic[1] = chunk_beg;
  }
  m->user_requested_size = size;
  StackTrace::CompressStack(stack, m->AllocStackBeg(), m->AllocStackSize());

  uptr size_rounded_down_to_granularity = RoundDownTo(size, SHADOW_GRANULARITY);
  // Unpoison the bulk of the memory region.
  if (size_rounded_down_to_granularity)
    PoisonShadow(user_beg, size_rounded_down_to_granularity, 0);
  // Deal with the end of the region if size is not aligned to granularity.
  if (size != size_rounded_down_to_granularity && flags()->poison_heap) {
    u8 *shadow = (u8*)MemToShadow(user_beg + size_rounded_down_to_granularity);
    *shadow = size & (SHADOW_GRANULARITY - 1);
  }

  AsanStats &thread_stats = asanThreadRegistry().GetCurrentThreadStats();
  thread_stats.mallocs++;
  thread_stats.malloced += size;
  thread_stats.malloced_redzones += needed_size - size;
  uptr class_id = Min(kNumberOfSizeClasses, SizeClassMap::ClassID(needed_size));
  thread_stats.malloced_by_size[class_id]++;
  if (needed_size > SizeClassMap::kMaxSize)
    thread_stats.malloc_large++;

  void *res = reinterpret_cast<void *>(user_beg);
  ASAN_MALLOC_HOOK(res, size);
  return res;
}

static void Deallocate(void *ptr, StackTrace *stack, AllocType alloc_type) {
  uptr p = reinterpret_cast<uptr>(ptr);
  if (p == 0 || p == kReturnOnZeroMalloc) return;
  uptr chunk_beg = p - kChunkHeaderSize;
  AsanChunk *m = reinterpret_cast<AsanChunk *>(chunk_beg);

  // Flip the chunk_state atomically to avoid race on double-free.
  u8 old_chunk_state = atomic_exchange((atomic_uint8_t*)m, CHUNK_QUARANTINE,
                                       memory_order_acq_rel);

  if (old_chunk_state == CHUNK_QUARANTINE)
    ReportDoubleFree((uptr)ptr, stack);
  else if (old_chunk_state != CHUNK_ALLOCATED)
    ReportFreeNotMalloced((uptr)ptr, stack);
  CHECK(old_chunk_state == CHUNK_ALLOCATED);
  if (m->alloc_type != alloc_type && flags()->alloc_dealloc_mismatch)
    ReportAllocTypeMismatch((uptr)ptr, stack,
                            (AllocType)m->alloc_type, (AllocType)alloc_type);

  CHECK_GE(m->alloc_tid, 0);
  if (SANITIZER_WORDSIZE == 64)  // On 32-bits this resides in user area.
    CHECK_EQ(m->free_tid, kInvalidTid);
  AsanThread *t = asanThreadRegistry().GetCurrent();
  m->free_tid = t ? t->tid() : 0;
  StackTrace::CompressStack(stack, m->FreeStackBeg(), m->FreeStackSize());
  CHECK(m->chunk_state == CHUNK_QUARANTINE);
  // Poison the region.
  PoisonShadow(m->Beg(),
               RoundUpTo(m->user_requested_size, SHADOW_GRANULARITY),
               kAsanHeapFreeMagic);

  AsanStats &thread_stats = asanThreadRegistry().GetCurrentThreadStats();
  thread_stats.frees++;
  thread_stats.freed += m->UsedSize();

  // Push into quarantine.
  if (t) {
    AsanChunkFifoList &q = t->malloc_storage().quarantine_;
    q.Push(m);

    if (q.size() > kMaxThreadLocalQuarantine)
      quarantine.SwallowThreadLocalQuarantine(&t->malloc_storage());
  } else {
    quarantine.BypassThreadLocalQuarantine(m);
  }

  ASAN_FREE_HOOK(ptr);
}

static void *Reallocate(void *old_ptr, uptr new_size, StackTrace *stack) {
  CHECK(old_ptr && new_size);
  uptr p = reinterpret_cast<uptr>(old_ptr);
  uptr chunk_beg = p - kChunkHeaderSize;
  AsanChunk *m = reinterpret_cast<AsanChunk *>(chunk_beg);

  AsanStats &thread_stats = asanThreadRegistry().GetCurrentThreadStats();
  thread_stats.reallocs++;
  thread_stats.realloced += new_size;

  CHECK(m->chunk_state == CHUNK_ALLOCATED);
  uptr old_size = m->UsedSize();
  uptr memcpy_size = Min(new_size, old_size);
  void *new_ptr = Allocate(new_size, 8, stack, FROM_MALLOC);
  if (new_ptr) {
    CHECK(REAL(memcpy) != 0);
    REAL(memcpy)(new_ptr, old_ptr, memcpy_size);
    Deallocate(old_ptr, stack, FROM_MALLOC);
  }
  return new_ptr;
}

static AsanChunk *GetAsanChunkByAddr(uptr p) {
  uptr alloc_beg = reinterpret_cast<uptr>(
      allocator.GetBlockBegin(reinterpret_cast<void *>(p)));
  if (!alloc_beg) return 0;
  uptr *memalign_magic = reinterpret_cast<uptr *>(alloc_beg);
  if (memalign_magic[0] == kMemalignMagic) {
      AsanChunk *m = reinterpret_cast<AsanChunk *>(memalign_magic[1]);
      CHECK(m->from_memalign);
      return m;
  }
  uptr chunk_beg = alloc_beg + ComputeRZSize(0) - kChunkHeaderSize;
  return reinterpret_cast<AsanChunk *>(chunk_beg);
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
  // Prefer an allocated chunk or a chunk from quarantine.
  if (left_chunk->chunk_state == CHUNK_AVAILABLE &&
      right_chunk->chunk_state != CHUNK_AVAILABLE)
    return right_chunk;
  if (right_chunk->chunk_state == CHUNK_AVAILABLE &&
      left_chunk->chunk_state != CHUNK_AVAILABLE)
    return left_chunk;
  // Choose based on offset.
  uptr l_offset = 0, r_offset = 0;
  CHECK(AsanChunkView(left_chunk).AddrIsAtRight(addr, 1, &l_offset));
  CHECK(AsanChunkView(right_chunk).AddrIsAtLeft(addr, 1, &r_offset));
  if (l_offset < r_offset)
    return left_chunk;
  return right_chunk;
}

AsanChunkView FindHeapChunkByAddress(uptr addr) {
  AsanChunk *m1 = GetAsanChunkByAddr(addr);
  if (!m1) return AsanChunkView(m1);
  uptr offset = 0;
  if (AsanChunkView(m1).AddrIsAtLeft(addr, 1, &offset)) {
    // The address is in the chunk's left redzone, so maybe it is actually
    // a right buffer overflow from the other chunk to the left.
    // Search a bit to the left to see if there is another chunk.
    AsanChunk *m2 = 0;
    for (uptr l = 1; l < GetPageSizeCached(); l++) {
      m2 = GetAsanChunkByAddr(addr - l);
      if (m2 == m1) continue;  // Still the same chunk.
      Printf("m1 %p m2 %p l %zd\n", m1, m2, l);
      break;
    }
    if (m2 && AsanChunkView(m2).AddrIsAtRight(addr, 1, &offset))
      m1 = ChooseChunk(addr, m2, m1);
  }
  return AsanChunkView(m1);
}

void AsanThreadLocalMallocStorage::CommitBack() {
  quarantine.SwallowThreadLocalQuarantine(this);
  allocator.SwallowCache(GetAllocatorCache(this));
}

SANITIZER_INTERFACE_ATTRIBUTE
void *asan_memalign(uptr alignment, uptr size, StackTrace *stack,
                    AllocType alloc_type) {
  return Allocate(size, alignment, stack, alloc_type);
}

SANITIZER_INTERFACE_ATTRIBUTE
void asan_free(void *ptr, StackTrace *stack, AllocType alloc_type) {
  Deallocate(ptr, stack, alloc_type);
}

SANITIZER_INTERFACE_ATTRIBUTE
void *asan_malloc(uptr size, StackTrace *stack) {
  return Allocate(size, 8, stack, FROM_MALLOC);
}

void *asan_calloc(uptr nmemb, uptr size, StackTrace *stack) {
  void *ptr = Allocate(nmemb * size, 8, stack, FROM_MALLOC);
  if (ptr)
    REAL(memset)(ptr, 0, nmemb * size);
  return ptr;
}

void *asan_realloc(void *p, uptr size, StackTrace *stack) {
  if (p == 0)
    return Allocate(size, 8, stack, FROM_MALLOC);
  if (size == 0) {
    Deallocate(p, stack, FROM_MALLOC);
    return 0;
  }
  return Reallocate(p, size, stack);
}

void *asan_valloc(uptr size, StackTrace *stack) {
  return Allocate(size, GetPageSizeCached(), stack, FROM_MALLOC);
}

void *asan_pvalloc(uptr size, StackTrace *stack) {
  uptr PageSize = GetPageSizeCached();
  size = RoundUpTo(size, PageSize);
  if (size == 0) {
    // pvalloc(0) should allocate one page.
    size = PageSize;
  }
  return Allocate(size, PageSize, stack, FROM_MALLOC);
}

int asan_posix_memalign(void **memptr, uptr alignment, uptr size,
                        StackTrace *stack) {
  void *ptr = Allocate(size, alignment, stack, FROM_MALLOC);
  CHECK(IsAligned((uptr)ptr, alignment));
  *memptr = ptr;
  return 0;
}

uptr asan_malloc_usable_size(void *ptr, StackTrace *stack) {
  CHECK(stack);
  if (ptr == 0) return 0;
  uptr usable_size = AllocationSize(reinterpret_cast<uptr>(ptr));
  if (flags()->check_malloc_usable_size && (usable_size == 0))
    ReportMallocUsableSizeNotOwned((uptr)ptr, stack);
  return usable_size;
}

uptr asan_mz_size(const void *ptr) {
  UNIMPLEMENTED();
  return 0;
}

void asan_mz_force_lock() {
  UNIMPLEMENTED();
}

void asan_mz_force_unlock() {
  UNIMPLEMENTED();
}

}  // namespace __asan

// ---------------------- Interface ---------------- {{{1
using namespace __asan;  // NOLINT

// ASan allocator doesn't reserve extra bytes, so normally we would
// just return "size". We don't want to expose our redzone sizes, etc here.
uptr __asan_get_estimated_allocated_size(uptr size) {
  return size;
}

bool __asan_get_ownership(const void *p) {
  return AllocationSize(reinterpret_cast<uptr>(p)) > 0;
}

uptr __asan_get_allocated_size(const void *p) {
  if (p == 0) return 0;
  uptr allocated_size = AllocationSize(reinterpret_cast<uptr>(p));
  // Die if p is not malloced or if it is already freed.
  if (allocated_size == 0) {
    GET_STACK_TRACE_FATAL_HERE;
    ReportAsanGetAllocatedSizeNotOwned(reinterpret_cast<uptr>(p), &stack);
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


#endif  // ASAN_ALLOCATOR_VERSION
