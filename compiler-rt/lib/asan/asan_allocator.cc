//===-- asan_allocator.cc -------------------------------------------------===//
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
// Implementation of ASan's memory allocator.
// Evey piece of memory (AsanChunk) allocated by the allocator
// has a left redzone of REDZONE bytes and
// a right redzone such that the end of the chunk is aligned by REDZONE
// (i.e. the right redzone is between 0 and REDZONE-1).
// The left redzone is always poisoned.
// The right redzone is poisoned on malloc, the body is poisoned on free.
// Once freed, a chunk is moved to a quarantine (fifo list).
// After quarantine, a chunk is returned to freelists.
//
// The left redzone contains ASan's internal data and the stack trace of
// the malloc call.
// Once freed, the body of the chunk contains the stack trace of the free call.
//
//===----------------------------------------------------------------------===//
#include "asan_allocator.h"

#if ASAN_ALLOCATOR_VERSION == 1
#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_stats.h"
#include "asan_report.h"
#include "asan_thread.h"
#include "asan_thread_registry.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_mutex.h"

namespace __asan {

#define REDZONE ((uptr)(flags()->redzone))
static const uptr kMinAllocSize = REDZONE * 2;
static const u64 kMaxAvailableRam = 128ULL << 30;  // 128G
static const uptr kMaxThreadLocalQuarantine = 1 << 20;  // 1M

static const uptr kMinMmapSize = (ASAN_LOW_MEMORY) ? 4UL << 17 : 4UL << 20;
static const uptr kMaxSizeForThreadLocalFreeList =
    (ASAN_LOW_MEMORY) ? 1 << 15 : 1 << 17;

// Size classes less than kMallocSizeClassStep are powers of two.
// All other size classes are multiples of kMallocSizeClassStep.
static const uptr kMallocSizeClassStepLog = 26;
static const uptr kMallocSizeClassStep = 1UL << kMallocSizeClassStepLog;

static const uptr kMaxAllowedMallocSize =
    (SANITIZER_WORDSIZE == 32) ? 3UL << 30 : 8UL << 30;

static inline uptr SizeClassToSize(u8 size_class) {
  CHECK(size_class < kNumberOfSizeClasses);
  if (size_class <= kMallocSizeClassStepLog) {
    return 1UL << size_class;
  } else {
    return (size_class - kMallocSizeClassStepLog) * kMallocSizeClassStep;
  }
}

static inline u8 SizeToSizeClass(uptr size) {
  u8 res = 0;
  if (size <= kMallocSizeClassStep) {
    uptr rounded = RoundUpToPowerOfTwo(size);
    res = Log2(rounded);
  } else {
    res = ((size + kMallocSizeClassStep - 1) / kMallocSizeClassStep)
        + kMallocSizeClassStepLog;
  }
  CHECK(res < kNumberOfSizeClasses);
  CHECK(size <= SizeClassToSize(res));
  return res;
}

// Given REDZONE bytes, we need to mark first size bytes
// as addressable and the rest REDZONE-size bytes as unaddressable.
static void PoisonHeapPartialRightRedzone(uptr mem, uptr size) {
  CHECK(size <= REDZONE);
  CHECK(IsAligned(mem, REDZONE));
  CHECK(IsPowerOfTwo(SHADOW_GRANULARITY));
  CHECK(IsPowerOfTwo(REDZONE));
  CHECK(REDZONE >= SHADOW_GRANULARITY);
  PoisonShadowPartialRightRedzone(mem, size, REDZONE,
                                  kAsanHeapRightRedzoneMagic);
}

static u8 *MmapNewPagesAndPoisonShadow(uptr size) {
  CHECK(IsAligned(size, GetPageSizeCached()));
  u8 *res = (u8*)MmapOrDie(size, __FUNCTION__);
  PoisonShadow((uptr)res, size, kAsanHeapLeftRedzoneMagic);
  if (flags()->debug) {
    Printf("ASAN_MMAP: [%p, %p)\n", res, res + size);
  }
  return res;
}

// Every chunk of memory allocated by this allocator can be in one of 3 states:
// CHUNK_AVAILABLE: the chunk is in the free list and ready to be allocated.
// CHUNK_ALLOCATED: the chunk is allocated and not yet freed.
// CHUNK_QUARANTINE: the chunk was freed and put into quarantine zone.
//
// The pseudo state CHUNK_MEMALIGN is used to mark that the address is not
// the beginning of a AsanChunk (in which the actual chunk resides at
// this - this->used_size).
//
// The magic numbers for the enum values are taken randomly.
enum {
  CHUNK_AVAILABLE  = 0x57,
  CHUNK_ALLOCATED  = 0x32,
  CHUNK_QUARANTINE = 0x19,
  CHUNK_MEMALIGN   = 0xDC
};

struct ChunkBase {
  // First 8 bytes.
  uptr  chunk_state : 8;
  uptr  alloc_tid   : 24;
  uptr  size_class  : 8;
  uptr  free_tid    : 24;

  // Second 8 bytes.
  uptr alignment_log : 8;
  uptr alloc_type    : 2;
  uptr used_size : FIRST_32_SECOND_64(32, 54);  // Size requested by the user.

  // This field may overlap with the user area and thus should not
  // be used while the chunk is in CHUNK_ALLOCATED state.
  AsanChunk *next;

  // Typically the beginning of the user-accessible memory is 'this'+REDZONE
  // and is also aligned by REDZONE. However, if the memory is allocated
  // by memalign, the alignment might be higher and the user-accessible memory
  // starts at the first properly aligned address after 'this'.
  uptr Beg() { return RoundUpTo((uptr)this + 1, 1 << alignment_log); }
  uptr Size() { return SizeClassToSize(size_class); }
  u8 SizeClass() { return size_class; }
};

struct AsanChunk: public ChunkBase {
  u32 *compressed_alloc_stack() {
    return (u32*)((uptr)this + sizeof(ChunkBase));
  }
  u32 *compressed_free_stack() {
    return (u32*)((uptr)this + Max((uptr)REDZONE, (uptr)sizeof(ChunkBase)));
  }

  // The left redzone after the ChunkBase is given to the alloc stack trace.
  uptr compressed_alloc_stack_size() {
    if (REDZONE < sizeof(ChunkBase)) return 0;
    return (REDZONE - sizeof(ChunkBase)) / sizeof(u32);
  }
  uptr compressed_free_stack_size() {
    if (REDZONE < sizeof(ChunkBase)) return 0;
    return (REDZONE) / sizeof(u32);
  }
};

uptr AsanChunkView::Beg() { return chunk_->Beg(); }
uptr AsanChunkView::End() { return Beg() + UsedSize(); }
uptr AsanChunkView::UsedSize() { return chunk_->used_size; }
uptr AsanChunkView::AllocTid() { return chunk_->alloc_tid; }
uptr AsanChunkView::FreeTid() { return chunk_->free_tid; }

void AsanChunkView::GetAllocStack(StackTrace *stack) {
  StackTrace::UncompressStack(stack, chunk_->compressed_alloc_stack(),
                              chunk_->compressed_alloc_stack_size());
}

void AsanChunkView::GetFreeStack(StackTrace *stack) {
  StackTrace::UncompressStack(stack, chunk_->compressed_free_stack(),
                              chunk_->compressed_free_stack_size());
}

static AsanChunk *PtrToChunk(uptr ptr) {
  AsanChunk *m = (AsanChunk*)(ptr - REDZONE);
  if (m->chunk_state == CHUNK_MEMALIGN) {
    m = (AsanChunk*)((uptr)m - m->used_size);
  }
  return m;
}

void AsanChunkFifoList::PushList(AsanChunkFifoList *q) {
  CHECK(q->size() > 0);
  size_ += q->size();
  append_back(q);
  q->clear();
}

void AsanChunkFifoList::Push(AsanChunk *n) {
  push_back(n);
  size_ += n->Size();
}

// Interesting performance observation: this function takes up to 15% of overal
// allocator time. That's because *first_ has been evicted from cache long time
// ago. Not sure if we can or want to do anything with this.
AsanChunk *AsanChunkFifoList::Pop() {
  CHECK(first_);
  AsanChunk *res = front();
  size_ -= res->Size();
  pop_front();
  return res;
}

// All pages we ever allocated.
struct PageGroup {
  uptr beg;
  uptr end;
  uptr size_of_chunk;
  uptr last_chunk;
  bool InRange(uptr addr) {
    return addr >= beg && addr < end;
  }
};

class MallocInfo {
 public:
  explicit MallocInfo(LinkerInitialized x) : mu_(x) { }

  AsanChunk *AllocateChunks(u8 size_class, uptr n_chunks) {
    AsanChunk *m = 0;
    AsanChunk **fl = &free_lists_[size_class];
    {
      BlockingMutexLock lock(&mu_);
      for (uptr i = 0; i < n_chunks; i++) {
        if (!(*fl)) {
          *fl = GetNewChunks(size_class);
        }
        AsanChunk *t = *fl;
        *fl = t->next;
        t->next = m;
        CHECK(t->chunk_state == CHUNK_AVAILABLE);
        m = t;
      }
    }
    return m;
  }

  void SwallowThreadLocalMallocStorage(AsanThreadLocalMallocStorage *x,
                                       bool eat_free_lists) {
    CHECK(flags()->quarantine_size > 0);
    BlockingMutexLock lock(&mu_);
    AsanChunkFifoList *q = &x->quarantine_;
    if (q->size() > 0) {
      quarantine_.PushList(q);
      while (quarantine_.size() > (uptr)flags()->quarantine_size) {
        QuarantinePop();
      }
    }
    if (eat_free_lists) {
      for (uptr size_class = 0; size_class < kNumberOfSizeClasses;
           size_class++) {
        AsanChunk *m = x->free_lists_[size_class];
        while (m) {
          AsanChunk *t = m->next;
          m->next = free_lists_[size_class];
          free_lists_[size_class] = m;
          m = t;
        }
        x->free_lists_[size_class] = 0;
      }
    }
  }

  void BypassThreadLocalQuarantine(AsanChunk *chunk) {
    BlockingMutexLock lock(&mu_);
    quarantine_.Push(chunk);
  }

  AsanChunk *FindChunkByAddr(uptr addr) {
    BlockingMutexLock lock(&mu_);
    return FindChunkByAddrUnlocked(addr);
  }

  uptr AllocationSize(uptr ptr) {
    if (!ptr) return 0;
    BlockingMutexLock lock(&mu_);

    // Make sure this is our chunk and |ptr| actually points to the beginning
    // of the allocated memory.
    AsanChunk *m = FindChunkByAddrUnlocked(ptr);
    if (!m || m->Beg() != ptr) return 0;

    if (m->chunk_state == CHUNK_ALLOCATED) {
      return m->used_size;
    } else {
      return 0;
    }
  }

  void ForceLock() {
    mu_.Lock();
  }

  void ForceUnlock() {
    mu_.Unlock();
  }

  void PrintStatus() {
    BlockingMutexLock lock(&mu_);
    uptr malloced = 0;

    Printf(" MallocInfo: in quarantine: %zu malloced: %zu; ",
           quarantine_.size() >> 20, malloced >> 20);
    for (uptr j = 1; j < kNumberOfSizeClasses; j++) {
      AsanChunk *i = free_lists_[j];
      if (!i) continue;
      uptr t = 0;
      for (; i; i = i->next) {
        t += i->Size();
      }
      Printf("%zu:%zu ", j, t >> 20);
    }
    Printf("\n");
  }

  PageGroup *FindPageGroup(uptr addr) {
    BlockingMutexLock lock(&mu_);
    return FindPageGroupUnlocked(addr);
  }

 private:
  PageGroup *FindPageGroupUnlocked(uptr addr) {
    int n = atomic_load(&n_page_groups_, memory_order_relaxed);
    // If the page groups are not sorted yet, sort them.
    if (n_sorted_page_groups_ < n) {
      SortArray((uptr*)page_groups_, n);
      n_sorted_page_groups_ = n;
    }
    // Binary search over the page groups.
    int beg = 0, end = n;
    while (beg < end) {
      int med = (beg + end) / 2;
      uptr g = (uptr)page_groups_[med];
      if (addr > g) {
        // 'g' points to the end of the group, so 'addr'
        // may not belong to page_groups_[med] or any previous group.
        beg = med + 1;
      } else {
        // 'addr' may belong to page_groups_[med] or a previous group.
        end = med;
      }
    }
    if (beg >= n)
      return 0;
    PageGroup *g = page_groups_[beg];
    CHECK(g);
    if (g->InRange(addr))
      return g;
    return 0;
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
    sptr l_offset = 0, r_offset = 0;
    CHECK(AsanChunkView(left_chunk).AddrIsAtRight(addr, 1, &l_offset));
    CHECK(AsanChunkView(right_chunk).AddrIsAtLeft(addr, 1, &r_offset));
    if (l_offset < r_offset)
      return left_chunk;
    return right_chunk;
  }

  AsanChunk *FindChunkByAddrUnlocked(uptr addr) {
    PageGroup *g = FindPageGroupUnlocked(addr);
    if (!g) return 0;
    CHECK(g->size_of_chunk);
    uptr offset_from_beg = addr - g->beg;
    uptr this_chunk_addr = g->beg +
        (offset_from_beg / g->size_of_chunk) * g->size_of_chunk;
    CHECK(g->InRange(this_chunk_addr));
    AsanChunk *m = (AsanChunk*)this_chunk_addr;
    CHECK(m->chunk_state == CHUNK_ALLOCATED ||
          m->chunk_state == CHUNK_AVAILABLE ||
          m->chunk_state == CHUNK_QUARANTINE);
    sptr offset = 0;
    AsanChunkView m_view(m);
    if (m_view.AddrIsInside(addr, 1, &offset))
      return m;

    if (m_view.AddrIsAtRight(addr, 1, &offset)) {
      if (this_chunk_addr == g->last_chunk)  // rightmost chunk
        return m;
      uptr right_chunk_addr = this_chunk_addr + g->size_of_chunk;
      CHECK(g->InRange(right_chunk_addr));
      return ChooseChunk(addr, m, (AsanChunk*)right_chunk_addr);
    } else {
      CHECK(m_view.AddrIsAtLeft(addr, 1, &offset));
      if (this_chunk_addr == g->beg)  // leftmost chunk
        return m;
      uptr left_chunk_addr = this_chunk_addr - g->size_of_chunk;
      CHECK(g->InRange(left_chunk_addr));
      return ChooseChunk(addr, (AsanChunk*)left_chunk_addr, m);
    }
  }

  void QuarantinePop() {
    CHECK(quarantine_.size() > 0);
    AsanChunk *m = quarantine_.Pop();
    CHECK(m);
    // if (F_v >= 2) Printf("MallocInfo::pop %p\n", m);

    CHECK(m->chunk_state == CHUNK_QUARANTINE);
    m->chunk_state = CHUNK_AVAILABLE;
    PoisonShadow((uptr)m, m->Size(), kAsanHeapLeftRedzoneMagic);
    CHECK(m->alloc_tid >= 0);
    CHECK(m->free_tid >= 0);

    uptr size_class = m->SizeClass();
    m->next = free_lists_[size_class];
    free_lists_[size_class] = m;

    // Statistics.
    AsanStats &thread_stats = asanThreadRegistry().GetCurrentThreadStats();
    thread_stats.real_frees++;
    thread_stats.really_freed += m->used_size;
    thread_stats.really_freed_redzones += m->Size() - m->used_size;
    thread_stats.really_freed_by_size[m->SizeClass()]++;
  }

  // Get a list of newly allocated chunks.
  AsanChunk *GetNewChunks(u8 size_class) {
    uptr size = SizeClassToSize(size_class);
    CHECK(IsPowerOfTwo(kMinMmapSize));
    CHECK(size < kMinMmapSize || (size % kMinMmapSize) == 0);
    uptr mmap_size = Max(size, kMinMmapSize);
    uptr n_chunks = mmap_size / size;
    CHECK(n_chunks * size == mmap_size);
    uptr PageSize = GetPageSizeCached();
    if (size < PageSize) {
      // Size is small, just poison the last chunk.
      n_chunks--;
    } else {
      // Size is large, allocate an extra page at right and poison it.
      mmap_size += PageSize;
    }
    CHECK(n_chunks > 0);
    u8 *mem = MmapNewPagesAndPoisonShadow(mmap_size);

    // Statistics.
    AsanStats &thread_stats = asanThreadRegistry().GetCurrentThreadStats();
    thread_stats.mmaps++;
    thread_stats.mmaped += mmap_size;
    thread_stats.mmaped_by_size[size_class] += n_chunks;

    AsanChunk *res = 0;
    for (uptr i = 0; i < n_chunks; i++) {
      AsanChunk *m = (AsanChunk*)(mem + i * size);
      m->chunk_state = CHUNK_AVAILABLE;
      m->size_class = size_class;
      m->next = res;
      res = m;
    }
    PageGroup *pg = (PageGroup*)(mem + n_chunks * size);
    // This memory is already poisoned, no need to poison it again.
    pg->beg = (uptr)mem;
    pg->end = pg->beg + mmap_size;
    pg->size_of_chunk = size;
    pg->last_chunk = (uptr)(mem + size * (n_chunks - 1));
    int idx = atomic_fetch_add(&n_page_groups_, 1, memory_order_relaxed);
    CHECK(idx < (int)ARRAY_SIZE(page_groups_));
    page_groups_[idx] = pg;
    return res;
  }

  AsanChunk *free_lists_[kNumberOfSizeClasses];
  AsanChunkFifoList quarantine_;
  BlockingMutex mu_;

  PageGroup *page_groups_[kMaxAvailableRam / kMinMmapSize];
  atomic_uint32_t n_page_groups_;
  int n_sorted_page_groups_;
};

static MallocInfo malloc_info(LINKER_INITIALIZED);

void AsanThreadLocalMallocStorage::CommitBack() {
  malloc_info.SwallowThreadLocalMallocStorage(this, true);
}

AsanChunkView FindHeapChunkByAddress(uptr address) {
  return AsanChunkView(malloc_info.FindChunkByAddr(address));
}

static u8 *Allocate(uptr alignment, uptr size, StackTrace *stack,
                    AllocType alloc_type) {
  __asan_init();
  CHECK(stack);
  if (size == 0) {
    size = 1;  // TODO(kcc): do something smarter
  }
  CHECK(IsPowerOfTwo(alignment));
  uptr rounded_size = RoundUpTo(size, REDZONE);
  uptr needed_size = rounded_size + REDZONE;
  if (alignment > REDZONE) {
    needed_size += alignment;
  }
  CHECK(IsAligned(needed_size, REDZONE));
  if (size > kMaxAllowedMallocSize || needed_size > kMaxAllowedMallocSize) {
    Report("WARNING: AddressSanitizer failed to allocate %p bytes\n",
           (void*)size);
    return 0;
  }

  u8 size_class = SizeToSizeClass(needed_size);
  uptr size_to_allocate = SizeClassToSize(size_class);
  CHECK(size_to_allocate >= kMinAllocSize);
  CHECK(size_to_allocate >= needed_size);
  CHECK(IsAligned(size_to_allocate, REDZONE));

  if (flags()->verbosity >= 3) {
    Printf("Allocate align: %zu size: %zu class: %u real: %zu\n",
         alignment, size, size_class, size_to_allocate);
  }

  AsanThread *t = GetCurrentThread();
  AsanStats &thread_stats = asanThreadRegistry().GetCurrentThreadStats();
  // Statistics
  thread_stats.mallocs++;
  thread_stats.malloced += size;
  thread_stats.malloced_redzones += size_to_allocate - size;
  thread_stats.malloced_by_size[size_class]++;

  AsanChunk *m = 0;
  if (!t || size_to_allocate >= kMaxSizeForThreadLocalFreeList) {
    // get directly from global storage.
    m = malloc_info.AllocateChunks(size_class, 1);
    thread_stats.malloc_large++;
  } else {
    // get from the thread-local storage.
    AsanChunk **fl = &t->malloc_storage().free_lists_[size_class];
    if (!*fl) {
      uptr n_new_chunks = kMaxSizeForThreadLocalFreeList / size_to_allocate;
      *fl = malloc_info.AllocateChunks(size_class, n_new_chunks);
      thread_stats.malloc_small_slow++;
    }
    m = *fl;
    *fl = (*fl)->next;
  }
  CHECK(m);
  CHECK(m->chunk_state == CHUNK_AVAILABLE);
  m->chunk_state = CHUNK_ALLOCATED;
  m->alloc_type = alloc_type;
  m->next = 0;
  CHECK(m->Size() == size_to_allocate);
  uptr addr = (uptr)m + REDZONE;
  CHECK(addr <= (uptr)m->compressed_free_stack());

  if (alignment > REDZONE && (addr & (alignment - 1))) {
    addr = RoundUpTo(addr, alignment);
    CHECK((addr & (alignment - 1)) == 0);
    AsanChunk *p = (AsanChunk*)(addr - REDZONE);
    p->chunk_state = CHUNK_MEMALIGN;
    p->used_size = (uptr)p - (uptr)m;
    m->alignment_log = Log2(alignment);
    CHECK(m->Beg() == addr);
  } else {
    m->alignment_log = Log2(REDZONE);
  }
  CHECK(m == PtrToChunk(addr));
  m->used_size = size;
  CHECK(m->Beg() == addr);
  m->alloc_tid = t ? t->tid() : 0;
  m->free_tid   = kInvalidTid;
  StackTrace::CompressStack(stack, m->compressed_alloc_stack(),
                                m->compressed_alloc_stack_size());
  PoisonShadow(addr, rounded_size, 0);
  if (size < rounded_size) {
    PoisonHeapPartialRightRedzone(addr + rounded_size - REDZONE,
                                  size & (REDZONE - 1));
  }
  if (size <= (uptr)(flags()->max_malloc_fill_size)) {
    REAL(memset)((void*)addr, 0, rounded_size);
  }
  return (u8*)addr;
}

static void Deallocate(u8 *ptr, StackTrace *stack, AllocType alloc_type) {
  if (!ptr) return;
  CHECK(stack);

  if (flags()->debug) {
    CHECK(malloc_info.FindPageGroup((uptr)ptr));
  }

  // Printf("Deallocate %p\n", ptr);
  AsanChunk *m = PtrToChunk((uptr)ptr);

  // Flip the chunk_state atomically to avoid race on double-free.
  u8 old_chunk_state = atomic_exchange((atomic_uint8_t*)m, CHUNK_QUARANTINE,
                                       memory_order_acq_rel);

  if (old_chunk_state == CHUNK_QUARANTINE) {
    ReportDoubleFree((uptr)ptr, stack);
  } else if (old_chunk_state != CHUNK_ALLOCATED) {
    ReportFreeNotMalloced((uptr)ptr, stack);
  }
  CHECK(old_chunk_state == CHUNK_ALLOCATED);
  if (m->alloc_type != alloc_type && flags()->alloc_dealloc_mismatch)
    ReportAllocTypeMismatch((uptr)ptr, stack,
                            (AllocType)m->alloc_type, (AllocType)alloc_type);
  // With REDZONE==16 m->next is in the user area, otherwise it should be 0.
  CHECK(REDZONE <= 16 || !m->next);
  CHECK(m->free_tid == kInvalidTid);
  CHECK(m->alloc_tid >= 0);
  AsanThread *t = GetCurrentThread();
  m->free_tid = t ? t->tid() : 0;
  StackTrace::CompressStack(stack, m->compressed_free_stack(),
                                m->compressed_free_stack_size());
  uptr rounded_size = RoundUpTo(m->used_size, REDZONE);
  PoisonShadow((uptr)ptr, rounded_size, kAsanHeapFreeMagic);

  // Statistics.
  AsanStats &thread_stats = asanThreadRegistry().GetCurrentThreadStats();
  thread_stats.frees++;
  thread_stats.freed += m->used_size;
  thread_stats.freed_by_size[m->SizeClass()]++;

  CHECK(m->chunk_state == CHUNK_QUARANTINE);

  if (t) {
    AsanThreadLocalMallocStorage *ms = &t->malloc_storage();
    ms->quarantine_.Push(m);

    if (ms->quarantine_.size() > kMaxThreadLocalQuarantine) {
      malloc_info.SwallowThreadLocalMallocStorage(ms, false);
    }
  } else {
    malloc_info.BypassThreadLocalQuarantine(m);
  }
}

static u8 *Reallocate(u8 *old_ptr, uptr new_size,
                           StackTrace *stack) {
  CHECK(old_ptr && new_size);

  // Statistics.
  AsanStats &thread_stats = asanThreadRegistry().GetCurrentThreadStats();
  thread_stats.reallocs++;
  thread_stats.realloced += new_size;

  AsanChunk *m = PtrToChunk((uptr)old_ptr);
  CHECK(m->chunk_state == CHUNK_ALLOCATED);
  uptr old_size = m->used_size;
  uptr memcpy_size = Min(new_size, old_size);
  u8 *new_ptr = Allocate(0, new_size, stack, FROM_MALLOC);
  if (new_ptr) {
    CHECK(REAL(memcpy) != 0);
    REAL(memcpy)(new_ptr, old_ptr, memcpy_size);
    Deallocate(old_ptr, stack, FROM_MALLOC);
  }
  return new_ptr;
}

}  // namespace __asan

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

namespace __asan {

void InitializeAllocator() { }

void PrintInternalAllocatorStats() {
}

SANITIZER_INTERFACE_ATTRIBUTE
void *asan_memalign(uptr alignment, uptr size, StackTrace *stack,
                    AllocType alloc_type) {
  void *ptr = (void*)Allocate(alignment, size, stack, alloc_type);
  ASAN_MALLOC_HOOK(ptr, size);
  return ptr;
}

SANITIZER_INTERFACE_ATTRIBUTE
void asan_free(void *ptr, StackTrace *stack, AllocType alloc_type) {
  ASAN_FREE_HOOK(ptr);
  Deallocate((u8*)ptr, stack, alloc_type);
}

SANITIZER_INTERFACE_ATTRIBUTE
void *asan_malloc(uptr size, StackTrace *stack) {
  void *ptr = (void*)Allocate(0, size, stack, FROM_MALLOC);
  ASAN_MALLOC_HOOK(ptr, size);
  return ptr;
}

void *asan_calloc(uptr nmemb, uptr size, StackTrace *stack) {
  if (__sanitizer::CallocShouldReturnNullDueToOverflow(size, nmemb)) return 0;
  void *ptr = (void*)Allocate(0, nmemb * size, stack, FROM_MALLOC);
  if (ptr)
    REAL(memset)(ptr, 0, nmemb * size);
  ASAN_MALLOC_HOOK(ptr, size);
  return ptr;
}

void *asan_realloc(void *p, uptr size, StackTrace *stack) {
  if (p == 0) {
    void *ptr = (void*)Allocate(0, size, stack, FROM_MALLOC);
    ASAN_MALLOC_HOOK(ptr, size);
    return ptr;
  } else if (size == 0) {
    ASAN_FREE_HOOK(p);
    Deallocate((u8*)p, stack, FROM_MALLOC);
    return 0;
  }
  return Reallocate((u8*)p, size, stack);
}

void *asan_valloc(uptr size, StackTrace *stack) {
  void *ptr = (void*)Allocate(GetPageSizeCached(), size, stack, FROM_MALLOC);
  ASAN_MALLOC_HOOK(ptr, size);
  return ptr;
}

void *asan_pvalloc(uptr size, StackTrace *stack) {
  uptr PageSize = GetPageSizeCached();
  size = RoundUpTo(size, PageSize);
  if (size == 0) {
    // pvalloc(0) should allocate one page.
    size = PageSize;
  }
  void *ptr = (void*)Allocate(PageSize, size, stack, FROM_MALLOC);
  ASAN_MALLOC_HOOK(ptr, size);
  return ptr;
}

int asan_posix_memalign(void **memptr, uptr alignment, uptr size,
                          StackTrace *stack) {
  void *ptr = Allocate(alignment, size, stack, FROM_MALLOC);
  CHECK(IsAligned((uptr)ptr, alignment));
  ASAN_MALLOC_HOOK(ptr, size);
  *memptr = ptr;
  return 0;
}

uptr asan_malloc_usable_size(void *ptr, StackTrace *stack) {
  CHECK(stack);
  if (ptr == 0) return 0;
  uptr usable_size = malloc_info.AllocationSize((uptr)ptr);
  if (flags()->check_malloc_usable_size && (usable_size == 0)) {
    ReportMallocUsableSizeNotOwned((uptr)ptr, stack);
  }
  return usable_size;
}

uptr asan_mz_size(const void *ptr) {
  return malloc_info.AllocationSize((uptr)ptr);
}

void asan_mz_force_lock() {
  malloc_info.ForceLock();
}

void asan_mz_force_unlock() {
  malloc_info.ForceUnlock();
}

}  // namespace __asan

// ---------------------- Interface ---------------- {{{1
using namespace __asan;  // NOLINT

// ASan allocator doesn't reserve extra bytes, so normally we would
// just return "size".
uptr __asan_get_estimated_allocated_size(uptr size) {
  if (size == 0) return 1;
  return Min(size, kMaxAllowedMallocSize);
}

bool __asan_get_ownership(const void *p) {
  return malloc_info.AllocationSize((uptr)p) > 0;
}

uptr __asan_get_allocated_size(const void *p) {
  if (p == 0) return 0;
  uptr allocated_size = malloc_info.AllocationSize((uptr)p);
  // Die if p is not malloced or if it is already freed.
  if (allocated_size == 0) {
    GET_STACK_TRACE_FATAL_HERE;
    ReportAsanGetAllocatedSizeNotOwned((uptr)p, &stack);
  }
  return allocated_size;
}
#endif  // ASAN_ALLOCATOR_VERSION
