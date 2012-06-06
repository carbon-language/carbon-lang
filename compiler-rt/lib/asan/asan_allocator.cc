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
#include "asan_interceptors.h"
#include "asan_interface.h"
#include "asan_internal.h"
#include "asan_lock.h"
#include "asan_mapping.h"
#include "asan_stats.h"
#include "asan_thread.h"
#include "asan_thread_registry.h"

#if defined(_WIN32) && !defined(__clang__)
#include <intrin.h>
#endif

namespace __asan {

#define  REDZONE FLAG_redzone
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
    (__WORDSIZE == 32) ? 3UL << 30 : 8UL << 30;

static inline bool IsAligned(uptr a, uptr alignment) {
  return (a & (alignment - 1)) == 0;
}

static inline uptr Log2(uptr x) {
  CHECK(IsPowerOfTwo(x));
#if !defined(_WIN32) || defined(__clang__)
  return __builtin_ctzl(x);
#elif defined(_WIN64)
  unsigned long ret;  // NOLINT
  _BitScanForward64(&ret, x);
  return ret;
#else
  unsigned long ret;  // NOLINT
  _BitScanForward(&ret, x);
  return ret;
#endif
}

static inline uptr RoundUpToPowerOfTwo(uptr size) {
  CHECK(size);
  if (IsPowerOfTwo(size)) return size;

  unsigned long up;  // NOLINT
#if !defined(_WIN32) || defined(__clang__)
  up = __WORDSIZE - 1 - __builtin_clzl(size);
#elif defined(_WIN64)
  _BitScanReverse64(&up, size);
#else
  _BitScanReverse(&up, size);
#endif
  CHECK(size < (1ULL << (up + 1)));
  CHECK(size > (1ULL << up));
  return 1UL << (up + 1);
}

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
  CHECK(IsAligned(size, kPageSize));
  u8 *res = (u8*)AsanMmapSomewhereOrDie(size, __FUNCTION__);
  PoisonShadow((uptr)res, size, kAsanHeapLeftRedzoneMagic);
  if (FLAG_debug) {
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
// the beginning of a AsanChunk (in which case 'next' contains the address
// of the AsanChunk).
//
// The magic numbers for the enum values are taken randomly.
enum {
  CHUNK_AVAILABLE  = 0x57,
  CHUNK_ALLOCATED  = 0x32,
  CHUNK_QUARANTINE = 0x19,
  CHUNK_MEMALIGN   = 0xDC,
};

struct ChunkBase {
  u8   chunk_state;
  u8   size_class;
  u32  offset;  // User-visible memory starts at this+offset (beg()).
  s32  alloc_tid;
  s32  free_tid;
  uptr used_size;  // Size requested by the user.
  AsanChunk *next;

  uptr   beg() { return (uptr)this + offset; }
  uptr Size() { return SizeClassToSize(size_class); }
  u8 SizeClass() { return size_class; }
};

struct AsanChunk: public ChunkBase {
  u32 *compressed_alloc_stack() {
    CHECK(REDZONE >= sizeof(ChunkBase));
    return (u32*)((uptr)this + sizeof(ChunkBase));
  }
  u32 *compressed_free_stack() {
    CHECK(REDZONE >= sizeof(ChunkBase));
    return (u32*)((uptr)this + REDZONE);
  }

  // The left redzone after the ChunkBase is given to the alloc stack trace.
  uptr compressed_alloc_stack_size() {
    return (REDZONE - sizeof(ChunkBase)) / sizeof(u32);
  }
  uptr compressed_free_stack_size() {
    return (REDZONE) / sizeof(u32);
  }

  bool AddrIsInside(uptr addr, uptr access_size, uptr *offset) {
    if (addr >= beg() && (addr + access_size) <= (beg() + used_size)) {
      *offset = addr - beg();
      return true;
    }
    return false;
  }

  bool AddrIsAtLeft(uptr addr, uptr access_size, uptr *offset) {
    if (addr < beg()) {
      *offset = beg() - addr;
      return true;
    }
    return false;
  }

  bool AddrIsAtRight(uptr addr, uptr access_size, uptr *offset) {
    if (addr + access_size >= beg() + used_size) {
      if (addr <= beg() + used_size)
        *offset = 0;
      else
        *offset = addr - (beg() + used_size);
      return true;
    }
    return false;
  }

  void DescribeAddress(uptr addr, uptr access_size) {
    uptr offset;
    AsanPrintf("%p is located ", (void*)addr);
    if (AddrIsInside(addr, access_size, &offset)) {
      AsanPrintf("%zu bytes inside of", offset);
    } else if (AddrIsAtLeft(addr, access_size, &offset)) {
      AsanPrintf("%zu bytes to the left of", offset);
    } else if (AddrIsAtRight(addr, access_size, &offset)) {
      AsanPrintf("%zu bytes to the right of", offset);
    } else {
      AsanPrintf(" somewhere around (this is AddressSanitizer bug!)");
    }
    AsanPrintf(" %zu-byte region [%p,%p)\n",
               used_size, (void*)beg(), (void*)(beg() + used_size));
  }
};

static AsanChunk *PtrToChunk(uptr ptr) {
  AsanChunk *m = (AsanChunk*)(ptr - REDZONE);
  if (m->chunk_state == CHUNK_MEMALIGN) {
    m = m->next;
  }
  return m;
}


void AsanChunkFifoList::PushList(AsanChunkFifoList *q) {
  CHECK(q->size() > 0);
  if (last_) {
    CHECK(first_);
    CHECK(!last_->next);
    last_->next = q->first_;
    last_ = q->last_;
  } else {
    CHECK(!first_);
    last_ = q->last_;
    first_ = q->first_;
    CHECK(first_);
  }
  CHECK(last_);
  CHECK(!last_->next);
  size_ += q->size();
  q->clear();
}

void AsanChunkFifoList::Push(AsanChunk *n) {
  CHECK(n->next == 0);
  if (last_) {
    CHECK(first_);
    CHECK(!last_->next);
    last_->next = n;
    last_ = n;
  } else {
    CHECK(!first_);
    last_ = first_ = n;
  }
  size_ += n->Size();
}

// Interesting performance observation: this function takes up to 15% of overal
// allocator time. That's because *first_ has been evicted from cache long time
// ago. Not sure if we can or want to do anything with this.
AsanChunk *AsanChunkFifoList::Pop() {
  CHECK(first_);
  AsanChunk *res = first_;
  first_ = first_->next;
  if (first_ == 0)
    last_ = 0;
  CHECK(size_ >= res->Size());
  size_ -= res->Size();
  if (last_) {
    CHECK(!last_->next);
  }
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
      ScopedLock lock(&mu_);
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
    CHECK(FLAG_quarantine_size > 0);
    ScopedLock lock(&mu_);
    AsanChunkFifoList *q = &x->quarantine_;
    if (q->size() > 0) {
      quarantine_.PushList(q);
      while (quarantine_.size() > FLAG_quarantine_size) {
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
    ScopedLock lock(&mu_);
    quarantine_.Push(chunk);
  }

  AsanChunk *FindMallocedOrFreed(uptr addr, uptr access_size) {
    ScopedLock lock(&mu_);
    return FindChunkByAddr(addr);
  }

  uptr AllocationSize(uptr ptr) {
    if (!ptr) return 0;
    ScopedLock lock(&mu_);

    // first, check if this is our memory
    PageGroup *g = FindPageGroupUnlocked(ptr);
    if (!g) return 0;
    AsanChunk *m = PtrToChunk(ptr);
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
    ScopedLock lock(&mu_);
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
    ScopedLock lock(&mu_);
    return FindPageGroupUnlocked(addr);
  }

 private:
  PageGroup *FindPageGroupUnlocked(uptr addr) {
    int n = n_page_groups_;
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
    uptr l_offset = 0, r_offset = 0;
    CHECK(left_chunk->AddrIsAtRight(addr, 1, &l_offset));
    CHECK(right_chunk->AddrIsAtLeft(addr, 1, &r_offset));
    if (l_offset < r_offset)
      return left_chunk;
    return right_chunk;
  }

  AsanChunk *FindChunkByAddr(uptr addr) {
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
    uptr offset = 0;
    if (m->AddrIsInside(addr, 1, &offset))
      return m;

    if (m->AddrIsAtRight(addr, 1, &offset)) {
      if (this_chunk_addr == g->last_chunk)  // rightmost chunk
        return m;
      uptr right_chunk_addr = this_chunk_addr + g->size_of_chunk;
      CHECK(g->InRange(right_chunk_addr));
      return ChooseChunk(addr, m, (AsanChunk*)right_chunk_addr);
    } else {
      CHECK(m->AddrIsAtLeft(addr, 1, &offset));
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
    if (size < kPageSize) {
      // Size is small, just poison the last chunk.
      n_chunks--;
    } else {
      // Size is large, allocate an extra page at right and poison it.
      mmap_size += kPageSize;
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
    int page_group_idx = AtomicInc(&n_page_groups_) - 1;
    CHECK(page_group_idx < (int)ASAN_ARRAY_SIZE(page_groups_));
    page_groups_[page_group_idx] = pg;
    return res;
  }

  AsanChunk *free_lists_[kNumberOfSizeClasses];
  AsanChunkFifoList quarantine_;
  AsanLock mu_;

  PageGroup *page_groups_[kMaxAvailableRam / kMinMmapSize];
  int n_page_groups_;  // atomic
  int n_sorted_page_groups_;
};

static MallocInfo malloc_info(LINKER_INITIALIZED);

void AsanThreadLocalMallocStorage::CommitBack() {
  malloc_info.SwallowThreadLocalMallocStorage(this, true);
}

static void Describe(uptr addr, uptr access_size) {
  AsanChunk *m = malloc_info.FindMallocedOrFreed(addr, access_size);
  if (!m) return;
  m->DescribeAddress(addr, access_size);
  CHECK(m->alloc_tid >= 0);
  AsanThreadSummary *alloc_thread =
      asanThreadRegistry().FindByTid(m->alloc_tid);
  AsanStackTrace alloc_stack;
  AsanStackTrace::UncompressStack(&alloc_stack, m->compressed_alloc_stack(),
                                  m->compressed_alloc_stack_size());
  AsanThread *t = asanThreadRegistry().GetCurrent();
  CHECK(t);
  if (m->free_tid >= 0) {
    AsanThreadSummary *free_thread =
        asanThreadRegistry().FindByTid(m->free_tid);
    AsanPrintf("freed by thread T%d here:\n", free_thread->tid());
    AsanStackTrace free_stack;
    AsanStackTrace::UncompressStack(&free_stack, m->compressed_free_stack(),
                                    m->compressed_free_stack_size());
    free_stack.PrintStack();
    AsanPrintf("previously allocated by thread T%d here:\n",
               alloc_thread->tid());

    alloc_stack.PrintStack();
    t->summary()->Announce();
    free_thread->Announce();
    alloc_thread->Announce();
  } else {
    AsanPrintf("allocated by thread T%d here:\n", alloc_thread->tid());
    alloc_stack.PrintStack();
    t->summary()->Announce();
    alloc_thread->Announce();
  }
}

static u8 *Allocate(uptr alignment, uptr size, AsanStackTrace *stack) {
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

  if (FLAG_v >= 3) {
    Printf("Allocate align: %zu size: %zu class: %u real: %zu\n",
         alignment, size, size_class, size_to_allocate);
  }

  AsanThread *t = asanThreadRegistry().GetCurrent();
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
  m->next = 0;
  CHECK(m->Size() == size_to_allocate);
  uptr addr = (uptr)m + REDZONE;
  CHECK(addr == (uptr)m->compressed_free_stack());

  if (alignment > REDZONE && (addr & (alignment - 1))) {
    addr = RoundUpTo(addr, alignment);
    CHECK((addr & (alignment - 1)) == 0);
    AsanChunk *p = (AsanChunk*)(addr - REDZONE);
    p->chunk_state = CHUNK_MEMALIGN;
    p->next = m;
  }
  CHECK(m == PtrToChunk(addr));
  m->used_size = size;
  m->offset = addr - (uptr)m;
  CHECK(m->beg() == addr);
  m->alloc_tid = t ? t->tid() : 0;
  m->free_tid   = AsanThread::kInvalidTid;
  AsanStackTrace::CompressStack(stack, m->compressed_alloc_stack(),
                                m->compressed_alloc_stack_size());
  PoisonShadow(addr, rounded_size, 0);
  if (size < rounded_size) {
    PoisonHeapPartialRightRedzone(addr + rounded_size - REDZONE,
                                  size & (REDZONE - 1));
  }
  if (size <= FLAG_max_malloc_fill_size) {
    REAL(memset)((void*)addr, 0, rounded_size);
  }
  return (u8*)addr;
}

static void Deallocate(u8 *ptr, AsanStackTrace *stack) {
  if (!ptr) return;
  CHECK(stack);

  if (FLAG_debug) {
    CHECK(malloc_info.FindPageGroup((uptr)ptr));
  }

  // Printf("Deallocate %p\n", ptr);
  AsanChunk *m = PtrToChunk((uptr)ptr);

  // Flip the state atomically to avoid race on double-free.
  u8 old_chunk_state = AtomicExchange(&m->chunk_state, CHUNK_QUARANTINE);

  if (old_chunk_state == CHUNK_QUARANTINE) {
    AsanReport("ERROR: AddressSanitizer attempting double-free on %p:\n", ptr);
    stack->PrintStack();
    Describe((uptr)ptr, 1);
    ShowStatsAndAbort();
  } else if (old_chunk_state != CHUNK_ALLOCATED) {
    AsanReport("ERROR: AddressSanitizer attempting free on address "
               "which was not malloc()-ed: %p\n", ptr);
    stack->PrintStack();
    ShowStatsAndAbort();
  }
  CHECK(old_chunk_state == CHUNK_ALLOCATED);
  CHECK(m->free_tid == AsanThread::kInvalidTid);
  CHECK(m->alloc_tid >= 0);
  AsanThread *t = asanThreadRegistry().GetCurrent();
  m->free_tid = t ? t->tid() : 0;
  AsanStackTrace::CompressStack(stack, m->compressed_free_stack(),
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
    CHECK(!m->next);
    ms->quarantine_.Push(m);

    if (ms->quarantine_.size() > kMaxThreadLocalQuarantine) {
      malloc_info.SwallowThreadLocalMallocStorage(ms, false);
    }
  } else {
    CHECK(!m->next);
    malloc_info.BypassThreadLocalQuarantine(m);
  }
}

static u8 *Reallocate(u8 *old_ptr, uptr new_size,
                           AsanStackTrace *stack) {
  CHECK(old_ptr && new_size);

  // Statistics.
  AsanStats &thread_stats = asanThreadRegistry().GetCurrentThreadStats();
  thread_stats.reallocs++;
  thread_stats.realloced += new_size;

  AsanChunk *m = PtrToChunk((uptr)old_ptr);
  CHECK(m->chunk_state == CHUNK_ALLOCATED);
  uptr old_size = m->used_size;
  uptr memcpy_size = Min(new_size, old_size);
  u8 *new_ptr = Allocate(0, new_size, stack);
  if (new_ptr) {
    CHECK(REAL(memcpy) != 0);
    REAL(memcpy)(new_ptr, old_ptr, memcpy_size);
    Deallocate(old_ptr, stack);
  }
  return new_ptr;
}

}  // namespace __asan

// Malloc hooks declaration.
// ASAN_NEW_HOOK(ptr, size) is called immediately after
//   allocation of "size" bytes, which returned "ptr".
// ASAN_DELETE_HOOK(ptr) is called immediately before
//   deallocation of "ptr".
// If ASAN_NEW_HOOK or ASAN_DELETE_HOOK is defined, user
// program must provide implementation of this hook.
// If macro is undefined, the hook is no-op.
#ifdef ASAN_NEW_HOOK
extern "C" void ASAN_NEW_HOOK(void *ptr, uptr size);
#else
static inline void ASAN_NEW_HOOK(void *ptr, uptr size) { }
#endif

#ifdef ASAN_DELETE_HOOK
extern "C" void ASAN_DELETE_HOOK(void *ptr);
#else
static inline void ASAN_DELETE_HOOK(void *ptr) { }
#endif

namespace __asan {

void *asan_memalign(uptr alignment, uptr size, AsanStackTrace *stack) {
  void *ptr = (void*)Allocate(alignment, size, stack);
  ASAN_NEW_HOOK(ptr, size);
  return ptr;
}

void asan_free(void *ptr, AsanStackTrace *stack) {
  ASAN_DELETE_HOOK(ptr);
  Deallocate((u8*)ptr, stack);
}

void *asan_malloc(uptr size, AsanStackTrace *stack) {
  void *ptr = (void*)Allocate(0, size, stack);
  ASAN_NEW_HOOK(ptr, size);
  return ptr;
}

void *asan_calloc(uptr nmemb, uptr size, AsanStackTrace *stack) {
  void *ptr = (void*)Allocate(0, nmemb * size, stack);
  if (ptr)
    REAL(memset)(ptr, 0, nmemb * size);
  ASAN_NEW_HOOK(ptr, nmemb * size);
  return ptr;
}

void *asan_realloc(void *p, uptr size, AsanStackTrace *stack) {
  if (p == 0) {
    void *ptr = (void*)Allocate(0, size, stack);
    ASAN_NEW_HOOK(ptr, size);
    return ptr;
  } else if (size == 0) {
    ASAN_DELETE_HOOK(p);
    Deallocate((u8*)p, stack);
    return 0;
  }
  return Reallocate((u8*)p, size, stack);
}

void *asan_valloc(uptr size, AsanStackTrace *stack) {
  void *ptr = (void*)Allocate(kPageSize, size, stack);
  ASAN_NEW_HOOK(ptr, size);
  return ptr;
}

void *asan_pvalloc(uptr size, AsanStackTrace *stack) {
  size = RoundUpTo(size, kPageSize);
  if (size == 0) {
    // pvalloc(0) should allocate one page.
    size = kPageSize;
  }
  void *ptr = (void*)Allocate(kPageSize, size, stack);
  ASAN_NEW_HOOK(ptr, size);
  return ptr;
}

int asan_posix_memalign(void **memptr, uptr alignment, uptr size,
                          AsanStackTrace *stack) {
  void *ptr = Allocate(alignment, size, stack);
  CHECK(IsAligned((uptr)ptr, alignment));
  ASAN_NEW_HOOK(ptr, size);
  *memptr = ptr;
  return 0;
}

uptr asan_malloc_usable_size(void *ptr, AsanStackTrace *stack) {
  CHECK(stack);
  if (ptr == 0) return 0;
  uptr usable_size = malloc_info.AllocationSize((uptr)ptr);
  if (FLAG_check_malloc_usable_size && (usable_size == 0)) {
    AsanReport("ERROR: AddressSanitizer attempting to call "
               "malloc_usable_size() for pointer which is "
               "not owned: %p\n", ptr);
    stack->PrintStack();
    Describe((uptr)ptr, 1);
    ShowStatsAndAbort();
  }
  return usable_size;
}

uptr asan_mz_size(const void *ptr) {
  return malloc_info.AllocationSize((uptr)ptr);
}

void DescribeHeapAddress(uptr addr, uptr access_size) {
  Describe(addr, access_size);
}

void asan_mz_force_lock() {
  malloc_info.ForceLock();
}

void asan_mz_force_unlock() {
  malloc_info.ForceUnlock();
}

// ---------------------- Fake stack-------------------- {{{1
FakeStack::FakeStack() {
  CHECK(REAL(memset) != 0);
  REAL(memset)(this, 0, sizeof(*this));
}

bool FakeStack::AddrIsInSizeClass(uptr addr, uptr size_class) {
  uptr mem = allocated_size_classes_[size_class];
  uptr size = ClassMmapSize(size_class);
  bool res = mem && addr >= mem && addr < mem + size;
  return res;
}

uptr FakeStack::AddrIsInFakeStack(uptr addr) {
  for (uptr i = 0; i < kNumberOfSizeClasses; i++) {
    if (AddrIsInSizeClass(addr, i)) return allocated_size_classes_[i];
  }
  return 0;
}

// We may want to compute this during compilation.
inline uptr FakeStack::ComputeSizeClass(uptr alloc_size) {
  uptr rounded_size = RoundUpToPowerOfTwo(alloc_size);
  uptr log = Log2(rounded_size);
  CHECK(alloc_size <= (1UL << log));
  if (!(alloc_size > (1UL << (log-1)))) {
    Printf("alloc_size %zu log %zu\n", alloc_size, log);
  }
  CHECK(alloc_size > (1UL << (log-1)));
  uptr res = log < kMinStackFrameSizeLog ? 0 : log - kMinStackFrameSizeLog;
  CHECK(res < kNumberOfSizeClasses);
  CHECK(ClassSize(res) >= rounded_size);
  return res;
}

void FakeFrameFifo::FifoPush(FakeFrame *node) {
  CHECK(node);
  node->next = 0;
  if (first_ == 0 && last_ == 0) {
    first_ = last_ = node;
  } else {
    CHECK(first_);
    CHECK(last_);
    last_->next = node;
    last_ = node;
  }
}

FakeFrame *FakeFrameFifo::FifoPop() {
  CHECK(first_ && last_ && "Exhausted fake stack");
  FakeFrame *res = 0;
  if (first_ == last_) {
    res = first_;
    first_ = last_ = 0;
  } else {
    res = first_;
    first_ = first_->next;
  }
  return res;
}

void FakeStack::Init(uptr stack_size) {
  stack_size_ = stack_size;
  alive_ = true;
}

void FakeStack::Cleanup() {
  alive_ = false;
  for (uptr i = 0; i < kNumberOfSizeClasses; i++) {
    uptr mem = allocated_size_classes_[i];
    if (mem) {
      PoisonShadow(mem, ClassMmapSize(i), 0);
      allocated_size_classes_[i] = 0;
      AsanUnmapOrDie((void*)mem, ClassMmapSize(i));
    }
  }
}

uptr FakeStack::ClassMmapSize(uptr size_class) {
  return RoundUpToPowerOfTwo(stack_size_);
}

void FakeStack::AllocateOneSizeClass(uptr size_class) {
  CHECK(ClassMmapSize(size_class) >= kPageSize);
  uptr new_mem = (uptr)AsanMmapSomewhereOrDie(
      ClassMmapSize(size_class), __FUNCTION__);
  // Printf("T%d new_mem[%zu]: %p-%p mmap %zu\n",
  //       asanThreadRegistry().GetCurrent()->tid(),
  //       size_class, new_mem, new_mem + ClassMmapSize(size_class),
  //       ClassMmapSize(size_class));
  uptr i;
  for (i = 0; i < ClassMmapSize(size_class);
       i += ClassSize(size_class)) {
    size_classes_[size_class].FifoPush((FakeFrame*)(new_mem + i));
  }
  CHECK(i == ClassMmapSize(size_class));
  allocated_size_classes_[size_class] = new_mem;
}

uptr FakeStack::AllocateStack(uptr size, uptr real_stack) {
  if (!alive_) return real_stack;
  CHECK(size <= kMaxStackMallocSize && size > 1);
  uptr size_class = ComputeSizeClass(size);
  if (!allocated_size_classes_[size_class]) {
    AllocateOneSizeClass(size_class);
  }
  FakeFrame *fake_frame = size_classes_[size_class].FifoPop();
  CHECK(fake_frame);
  fake_frame->size_minus_one = size - 1;
  fake_frame->real_stack = real_stack;
  while (FakeFrame *top = call_stack_.top()) {
    if (top->real_stack > real_stack) break;
    call_stack_.LifoPop();
    DeallocateFrame(top);
  }
  call_stack_.LifoPush(fake_frame);
  uptr ptr = (uptr)fake_frame;
  PoisonShadow(ptr, size, 0);
  return ptr;
}

void FakeStack::DeallocateFrame(FakeFrame *fake_frame) {
  CHECK(alive_);
  uptr size = fake_frame->size_minus_one + 1;
  uptr size_class = ComputeSizeClass(size);
  CHECK(allocated_size_classes_[size_class]);
  uptr ptr = (uptr)fake_frame;
  CHECK(AddrIsInSizeClass(ptr, size_class));
  CHECK(AddrIsInSizeClass(ptr + size - 1, size_class));
  size_classes_[size_class].FifoPush(fake_frame);
}

void FakeStack::OnFree(uptr ptr, uptr size, uptr real_stack) {
  FakeFrame *fake_frame = (FakeFrame*)ptr;
  CHECK(fake_frame->magic = kRetiredStackFrameMagic);
  CHECK(fake_frame->descr != 0);
  CHECK(fake_frame->size_minus_one == size - 1);
  PoisonShadow(ptr, size, kAsanStackAfterReturnMagic);
}

}  // namespace __asan

// ---------------------- Interface ---------------- {{{1
using namespace __asan;  // NOLINT

uptr __asan_stack_malloc(uptr size, uptr real_stack) {
  if (!FLAG_use_fake_stack) return real_stack;
  AsanThread *t = asanThreadRegistry().GetCurrent();
  if (!t) {
    // TSD is gone, use the real stack.
    return real_stack;
  }
  uptr ptr = t->fake_stack().AllocateStack(size, real_stack);
  // Printf("__asan_stack_malloc %p %zu %p\n", ptr, size, real_stack);
  return ptr;
}

void __asan_stack_free(uptr ptr, uptr size, uptr real_stack) {
  if (!FLAG_use_fake_stack) return;
  if (ptr != real_stack) {
    FakeStack::OnFree(ptr, size, real_stack);
  }
}

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
    AsanReport("ERROR: AddressSanitizer attempting to call "
               "__asan_get_allocated_size() for pointer which is "
               "not owned: %p\n", p);
    PRINT_CURRENT_STACK();
    Describe((uptr)p, 1);
    ShowStatsAndAbort();
  }
  return allocated_size;
}
