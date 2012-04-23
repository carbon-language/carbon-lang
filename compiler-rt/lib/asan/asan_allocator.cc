//===-- asan_allocator.cc ---------------------------------------*- C++ -*-===//
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

#ifdef _WIN32
#include <intrin.h>
#endif

namespace __asan {

#define  REDZONE FLAG_redzone
static const size_t kMinAllocSize = REDZONE * 2;
static const uint64_t kMaxAvailableRam = 128ULL << 30;  // 128G
static const size_t kMaxThreadLocalQuarantine = 1 << 20;  // 1M

static const size_t kMinMmapSize = (ASAN_LOW_MEMORY) ? 4UL << 17 : 4UL << 20;
static const size_t kMaxSizeForThreadLocalFreeList =
    (ASAN_LOW_MEMORY) ? 1 << 15 : 1 << 17;

// Size classes less than kMallocSizeClassStep are powers of two.
// All other size classes are multiples of kMallocSizeClassStep.
static const size_t kMallocSizeClassStepLog = 26;
static const size_t kMallocSizeClassStep = 1UL << kMallocSizeClassStepLog;

static const size_t kMaxAllowedMallocSize =
    (__WORDSIZE == 32) ? 3UL << 30 : 8UL << 30;

static inline bool IsAligned(uintptr_t a, uintptr_t alignment) {
  return (a & (alignment - 1)) == 0;
}

static inline size_t Log2(size_t x) {
  CHECK(IsPowerOfTwo(x));
#if defined(_WIN64)
  unsigned long ret;  // NOLINT
  _BitScanForward64(&ret, x);
  return ret;
#elif defined(_WIN32)
  unsigned long ret;  // NOLINT
  _BitScanForward(&ret, x);
  return ret;
#else
  return __builtin_ctzl(x);
#endif
}

static inline size_t RoundUpToPowerOfTwo(size_t size) {
  CHECK(size);
  if (IsPowerOfTwo(size)) return size;

  unsigned long up;  // NOLINT
#if defined(_WIN64)
  _BitScanReverse64(&up, size);
#elif defined(_WIN32)
  _BitScanReverse(&up, size);
#else
  up = __WORDSIZE - 1 - __builtin_clzl(size);
#endif
  CHECK(size < (1ULL << (up + 1)));
  CHECK(size > (1ULL << up));
  return 1UL << (up + 1);
}

static inline size_t SizeClassToSize(uint8_t size_class) {
  CHECK(size_class < kNumberOfSizeClasses);
  if (size_class <= kMallocSizeClassStepLog) {
    return 1UL << size_class;
  } else {
    return (size_class - kMallocSizeClassStepLog) * kMallocSizeClassStep;
  }
}

static inline uint8_t SizeToSizeClass(size_t size) {
  uint8_t res = 0;
  if (size <= kMallocSizeClassStep) {
    size_t rounded = RoundUpToPowerOfTwo(size);
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
static void PoisonHeapPartialRightRedzone(uintptr_t mem, size_t size) {
  CHECK(size <= REDZONE);
  CHECK(IsAligned(mem, REDZONE));
  CHECK(IsPowerOfTwo(SHADOW_GRANULARITY));
  CHECK(IsPowerOfTwo(REDZONE));
  CHECK(REDZONE >= SHADOW_GRANULARITY);
  PoisonShadowPartialRightRedzone(mem, size, REDZONE,
                                  kAsanHeapRightRedzoneMagic);
}

static uint8_t *MmapNewPagesAndPoisonShadow(size_t size) {
  CHECK(IsAligned(size, kPageSize));
  uint8_t *res = (uint8_t*)AsanMmapSomewhereOrDie(size, __FUNCTION__);
  PoisonShadow((uintptr_t)res, size, kAsanHeapLeftRedzoneMagic);
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
  CHUNK_AVAILABLE  = 0x573B,
  CHUNK_ALLOCATED  = 0x3204,
  CHUNK_QUARANTINE = 0x1978,
  CHUNK_MEMALIGN   = 0xDC68,
};

struct ChunkBase {
  uint16_t   chunk_state;
  uint8_t    size_class;
  uint32_t   offset;  // User-visible memory starts at this+offset (beg()).
  int32_t    alloc_tid;
  int32_t    free_tid;
  size_t     used_size;  // Size requested by the user.
  AsanChunk *next;

  uintptr_t   beg() { return (uintptr_t)this + offset; }
  size_t Size() { return SizeClassToSize(size_class); }
  uint8_t SizeClass() { return size_class; }
};

struct AsanChunk: public ChunkBase {
  uint32_t *compressed_alloc_stack() {
    CHECK(REDZONE >= sizeof(ChunkBase));
    return (uint32_t*)((uintptr_t)this + sizeof(ChunkBase));
  }
  uint32_t *compressed_free_stack() {
    CHECK(REDZONE >= sizeof(ChunkBase));
    return (uint32_t*)((uintptr_t)this + REDZONE);
  }

  // The left redzone after the ChunkBase is given to the alloc stack trace.
  size_t compressed_alloc_stack_size() {
    return (REDZONE - sizeof(ChunkBase)) / sizeof(uint32_t);
  }
  size_t compressed_free_stack_size() {
    return (REDZONE) / sizeof(uint32_t);
  }

  bool AddrIsInside(uintptr_t addr, size_t access_size, size_t *offset) {
    if (addr >= beg() && (addr + access_size) <= (beg() + used_size)) {
      *offset = addr - beg();
      return true;
    }
    return false;
  }

  bool AddrIsAtLeft(uintptr_t addr, size_t access_size, size_t *offset) {
    if (addr < beg()) {
      *offset = beg() - addr;
      return true;
    }
    return false;
  }

  bool AddrIsAtRight(uintptr_t addr, size_t access_size, size_t *offset) {
    if (addr + access_size >= beg() + used_size) {
      if (addr <= beg() + used_size)
        *offset = 0;
      else
        *offset = addr - (beg() + used_size);
      return true;
    }
    return false;
  }

  void DescribeAddress(uintptr_t addr, size_t access_size) {
    size_t offset;
    Printf("%p is located ", addr);
    if (AddrIsInside(addr, access_size, &offset)) {
      Printf("%zu bytes inside of", offset);
    } else if (AddrIsAtLeft(addr, access_size, &offset)) {
      Printf("%zu bytes to the left of", offset);
    } else if (AddrIsAtRight(addr, access_size, &offset)) {
      Printf("%zu bytes to the right of", offset);
    } else {
      Printf(" somewhere around (this is AddressSanitizer bug!)");
    }
    Printf(" %zu-byte region [%p,%p)\n",
           used_size, beg(), beg() + used_size);
  }
};

static AsanChunk *PtrToChunk(uintptr_t ptr) {
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
  CHECK(n->next == NULL);
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
  if (first_ == NULL)
    last_ = NULL;
  CHECK(size_ >= res->Size());
  size_ -= res->Size();
  if (last_) {
    CHECK(!last_->next);
  }
  return res;
}

// All pages we ever allocated.
struct PageGroup {
  uintptr_t beg;
  uintptr_t end;
  size_t size_of_chunk;
  uintptr_t last_chunk;
  bool InRange(uintptr_t addr) {
    return addr >= beg && addr < end;
  }
};

class MallocInfo {
 public:

  explicit MallocInfo(LinkerInitialized x) : mu_(x) { }

  AsanChunk *AllocateChunks(uint8_t size_class, size_t n_chunks) {
    AsanChunk *m = NULL;
    AsanChunk **fl = &free_lists_[size_class];
    {
      ScopedLock lock(&mu_);
      for (size_t i = 0; i < n_chunks; i++) {
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
      for (size_t size_class = 0; size_class < kNumberOfSizeClasses;
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

  AsanChunk *FindMallocedOrFreed(uintptr_t addr, size_t access_size) {
    ScopedLock lock(&mu_);
    return FindChunkByAddr(addr);
  }

  size_t AllocationSize(uintptr_t ptr) {
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
    size_t malloced = 0;

    Printf(" MallocInfo: in quarantine: %zu malloced: %zu; ",
           quarantine_.size() >> 20, malloced >> 20);
    for (size_t j = 1; j < kNumberOfSizeClasses; j++) {
      AsanChunk *i = free_lists_[j];
      if (!i) continue;
      size_t t = 0;
      for (; i; i = i->next) {
        t += i->Size();
      }
      Printf("%zu:%zu ", j, t >> 20);
    }
    Printf("\n");
  }

  PageGroup *FindPageGroup(uintptr_t addr) {
    ScopedLock lock(&mu_);
    return FindPageGroupUnlocked(addr);
  }

 private:
  PageGroup *FindPageGroupUnlocked(uintptr_t addr) {
    int n = n_page_groups_;
    // If the page groups are not sorted yet, sort them.
    if (n_sorted_page_groups_ < n) {
      SortArray((uintptr_t*)page_groups_, n);
      n_sorted_page_groups_ = n;
    }
    // Binary search over the page groups.
    int beg = 0, end = n;
    while (beg < end) {
      int med = (beg + end) / 2;
      uintptr_t g = (uintptr_t)page_groups_[med];
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
      return NULL;
    PageGroup *g = page_groups_[beg];
    CHECK(g);
    if (g->InRange(addr))
      return g;
    return NULL;
  }

  // We have an address between two chunks, and we want to report just one.
  AsanChunk *ChooseChunk(uintptr_t addr,
                         AsanChunk *left_chunk, AsanChunk *right_chunk) {
    // Prefer an allocated chunk or a chunk from quarantine.
    if (left_chunk->chunk_state == CHUNK_AVAILABLE &&
        right_chunk->chunk_state != CHUNK_AVAILABLE)
      return right_chunk;
    if (right_chunk->chunk_state == CHUNK_AVAILABLE &&
        left_chunk->chunk_state != CHUNK_AVAILABLE)
      return left_chunk;
    // Choose based on offset.
    size_t l_offset = 0, r_offset = 0;
    CHECK(left_chunk->AddrIsAtRight(addr, 1, &l_offset));
    CHECK(right_chunk->AddrIsAtLeft(addr, 1, &r_offset));
    if (l_offset < r_offset)
      return left_chunk;
    return right_chunk;
  }

  AsanChunk *FindChunkByAddr(uintptr_t addr) {
    PageGroup *g = FindPageGroupUnlocked(addr);
    if (!g) return 0;
    CHECK(g->size_of_chunk);
    uintptr_t offset_from_beg = addr - g->beg;
    uintptr_t this_chunk_addr = g->beg +
        (offset_from_beg / g->size_of_chunk) * g->size_of_chunk;
    CHECK(g->InRange(this_chunk_addr));
    AsanChunk *m = (AsanChunk*)this_chunk_addr;
    CHECK(m->chunk_state == CHUNK_ALLOCATED ||
          m->chunk_state == CHUNK_AVAILABLE ||
          m->chunk_state == CHUNK_QUARANTINE);
    size_t offset = 0;
    if (m->AddrIsInside(addr, 1, &offset))
      return m;

    if (m->AddrIsAtRight(addr, 1, &offset)) {
      if (this_chunk_addr == g->last_chunk)  // rightmost chunk
        return m;
      uintptr_t right_chunk_addr = this_chunk_addr + g->size_of_chunk;
      CHECK(g->InRange(right_chunk_addr));
      return ChooseChunk(addr, m, (AsanChunk*)right_chunk_addr);
    } else {
      CHECK(m->AddrIsAtLeft(addr, 1, &offset));
      if (this_chunk_addr == g->beg)  // leftmost chunk
        return m;
      uintptr_t left_chunk_addr = this_chunk_addr - g->size_of_chunk;
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
    PoisonShadow((uintptr_t)m, m->Size(), kAsanHeapLeftRedzoneMagic);
    CHECK(m->alloc_tid >= 0);
    CHECK(m->free_tid >= 0);

    size_t size_class = m->SizeClass();
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
  AsanChunk *GetNewChunks(uint8_t size_class) {
    size_t size = SizeClassToSize(size_class);
    CHECK(IsPowerOfTwo(kMinMmapSize));
    CHECK(size < kMinMmapSize || (size % kMinMmapSize) == 0);
    size_t mmap_size = Max(size, kMinMmapSize);
    size_t n_chunks = mmap_size / size;
    CHECK(n_chunks * size == mmap_size);
    if (size < kPageSize) {
      // Size is small, just poison the last chunk.
      n_chunks--;
    } else {
      // Size is large, allocate an extra page at right and poison it.
      mmap_size += kPageSize;
    }
    CHECK(n_chunks > 0);
    uint8_t *mem = MmapNewPagesAndPoisonShadow(mmap_size);

    // Statistics.
    AsanStats &thread_stats = asanThreadRegistry().GetCurrentThreadStats();
    thread_stats.mmaps++;
    thread_stats.mmaped += mmap_size;
    thread_stats.mmaped_by_size[size_class] += n_chunks;

    AsanChunk *res = NULL;
    for (size_t i = 0; i < n_chunks; i++) {
      AsanChunk *m = (AsanChunk*)(mem + i * size);
      m->chunk_state = CHUNK_AVAILABLE;
      m->size_class = size_class;
      m->next = res;
      res = m;
    }
    PageGroup *pg = (PageGroup*)(mem + n_chunks * size);
    // This memory is already poisoned, no need to poison it again.
    pg->beg = (uintptr_t)mem;
    pg->end = pg->beg + mmap_size;
    pg->size_of_chunk = size;
    pg->last_chunk = (uintptr_t)(mem + size * (n_chunks - 1));
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

static void Describe(uintptr_t addr, size_t access_size) {
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
    Printf("freed by thread T%d here:\n", free_thread->tid());
    AsanStackTrace free_stack;
    AsanStackTrace::UncompressStack(&free_stack, m->compressed_free_stack(),
                                    m->compressed_free_stack_size());
    free_stack.PrintStack();
    Printf("previously allocated by thread T%d here:\n",
           alloc_thread->tid());

    alloc_stack.PrintStack();
    t->summary()->Announce();
    free_thread->Announce();
    alloc_thread->Announce();
  } else {
    Printf("allocated by thread T%d here:\n", alloc_thread->tid());
    alloc_stack.PrintStack();
    t->summary()->Announce();
    alloc_thread->Announce();
  }
}

static uint8_t *Allocate(size_t alignment, size_t size, AsanStackTrace *stack) {
  __asan_init();
  CHECK(stack);
  if (size == 0) {
    size = 1;  // TODO(kcc): do something smarter
  }
  CHECK(IsPowerOfTwo(alignment));
  size_t rounded_size = RoundUpTo(size, REDZONE);
  size_t needed_size = rounded_size + REDZONE;
  if (alignment > REDZONE) {
    needed_size += alignment;
  }
  CHECK(IsAligned(needed_size, REDZONE));
  if (size > kMaxAllowedMallocSize || needed_size > kMaxAllowedMallocSize) {
    Report("WARNING: AddressSanitizer failed to allocate %p bytes\n", size);
    return 0;
  }

  uint8_t size_class = SizeToSizeClass(needed_size);
  size_t size_to_allocate = SizeClassToSize(size_class);
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

  AsanChunk *m = NULL;
  if (!t || size_to_allocate >= kMaxSizeForThreadLocalFreeList) {
    // get directly from global storage.
    m = malloc_info.AllocateChunks(size_class, 1);
    thread_stats.malloc_large++;
  } else {
    // get from the thread-local storage.
    AsanChunk **fl = &t->malloc_storage().free_lists_[size_class];
    if (!*fl) {
      size_t n_new_chunks = kMaxSizeForThreadLocalFreeList / size_to_allocate;
      *fl = malloc_info.AllocateChunks(size_class, n_new_chunks);
      thread_stats.malloc_small_slow++;
    }
    m = *fl;
    *fl = (*fl)->next;
  }
  CHECK(m);
  CHECK(m->chunk_state == CHUNK_AVAILABLE);
  m->chunk_state = CHUNK_ALLOCATED;
  m->next = NULL;
  CHECK(m->Size() == size_to_allocate);
  uintptr_t addr = (uintptr_t)m + REDZONE;
  CHECK(addr == (uintptr_t)m->compressed_free_stack());

  if (alignment > REDZONE && (addr & (alignment - 1))) {
    addr = RoundUpTo(addr, alignment);
    CHECK((addr & (alignment - 1)) == 0);
    AsanChunk *p = (AsanChunk*)(addr - REDZONE);
    p->chunk_state = CHUNK_MEMALIGN;
    p->next = m;
  }
  CHECK(m == PtrToChunk(addr));
  m->used_size = size;
  m->offset = addr - (uintptr_t)m;
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
  return (uint8_t*)addr;
}

static void Deallocate(uint8_t *ptr, AsanStackTrace *stack) {
  if (!ptr) return;
  CHECK(stack);

  if (FLAG_debug) {
    CHECK(malloc_info.FindPageGroup((uintptr_t)ptr));
  }

  // Printf("Deallocate %p\n", ptr);
  AsanChunk *m = PtrToChunk((uintptr_t)ptr);

  // Flip the state atomically to avoid race on double-free.
  uint16_t old_chunk_state = AtomicExchange(&m->chunk_state, CHUNK_QUARANTINE);

  if (old_chunk_state == CHUNK_QUARANTINE) {
    Report("ERROR: AddressSanitizer attempting double-free on %p:\n", ptr);
    stack->PrintStack();
    Describe((uintptr_t)ptr, 1);
    ShowStatsAndAbort();
  } else if (old_chunk_state != CHUNK_ALLOCATED) {
    Report("ERROR: AddressSanitizer attempting free on address which was not"
           " malloc()-ed: %p\n", ptr);
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
  size_t rounded_size = RoundUpTo(m->used_size, REDZONE);
  PoisonShadow((uintptr_t)ptr, rounded_size, kAsanHeapFreeMagic);

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

static uint8_t *Reallocate(uint8_t *old_ptr, size_t new_size,
                           AsanStackTrace *stack) {
  CHECK(old_ptr && new_size);

  // Statistics.
  AsanStats &thread_stats = asanThreadRegistry().GetCurrentThreadStats();
  thread_stats.reallocs++;
  thread_stats.realloced += new_size;

  AsanChunk *m = PtrToChunk((uintptr_t)old_ptr);
  CHECK(m->chunk_state == CHUNK_ALLOCATED);
  size_t old_size = m->used_size;
  size_t memcpy_size = Min(new_size, old_size);
  uint8_t *new_ptr = Allocate(0, new_size, stack);
  if (new_ptr) {
    CHECK(REAL(memcpy) != NULL);
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
extern "C" void ASAN_NEW_HOOK(void *ptr, size_t size);
#else
static inline void ASAN_NEW_HOOK(void *ptr, size_t size) { }
#endif

#ifdef ASAN_DELETE_HOOK
extern "C" void ASAN_DELETE_HOOK(void *ptr);
#else
static inline void ASAN_DELETE_HOOK(void *ptr) { }
#endif

namespace __asan {

void *asan_memalign(size_t alignment, size_t size, AsanStackTrace *stack) {
  void *ptr = (void*)Allocate(alignment, size, stack);
  ASAN_NEW_HOOK(ptr, size);
  return ptr;
}

void asan_free(void *ptr, AsanStackTrace *stack) {
  ASAN_DELETE_HOOK(ptr);
  Deallocate((uint8_t*)ptr, stack);
}

void *asan_malloc(size_t size, AsanStackTrace *stack) {
  void *ptr = (void*)Allocate(0, size, stack);
  ASAN_NEW_HOOK(ptr, size);
  return ptr;
}

void *asan_calloc(size_t nmemb, size_t size, AsanStackTrace *stack) {
  void *ptr = (void*)Allocate(0, nmemb * size, stack);
  if (ptr)
    REAL(memset)(ptr, 0, nmemb * size);
  ASAN_NEW_HOOK(ptr, nmemb * size);
  return ptr;
}

void *asan_realloc(void *p, size_t size, AsanStackTrace *stack) {
  if (p == NULL) {
    void *ptr = (void*)Allocate(0, size, stack);
    ASAN_NEW_HOOK(ptr, size);
    return ptr;
  } else if (size == 0) {
    ASAN_DELETE_HOOK(p);
    Deallocate((uint8_t*)p, stack);
    return NULL;
  }
  return Reallocate((uint8_t*)p, size, stack);
}

void *asan_valloc(size_t size, AsanStackTrace *stack) {
  void *ptr = (void*)Allocate(kPageSize, size, stack);
  ASAN_NEW_HOOK(ptr, size);
  return ptr;
}

void *asan_pvalloc(size_t size, AsanStackTrace *stack) {
  size = RoundUpTo(size, kPageSize);
  if (size == 0) {
    // pvalloc(0) should allocate one page.
    size = kPageSize;
  }
  void *ptr = (void*)Allocate(kPageSize, size, stack);
  ASAN_NEW_HOOK(ptr, size);
  return ptr;
}

int asan_posix_memalign(void **memptr, size_t alignment, size_t size,
                          AsanStackTrace *stack) {
  void *ptr = Allocate(alignment, size, stack);
  CHECK(IsAligned((uintptr_t)ptr, alignment));
  ASAN_NEW_HOOK(ptr, size);
  *memptr = ptr;
  return 0;
}

size_t asan_malloc_usable_size(void *ptr, AsanStackTrace *stack) {
  CHECK(stack);
  if (ptr == NULL) return 0;
  size_t usable_size = malloc_info.AllocationSize((uintptr_t)ptr);
  if (usable_size == 0) {
    Report("ERROR: AddressSanitizer attempting to call malloc_usable_size() "
           "for pointer which is not owned: %p\n", ptr);
    stack->PrintStack();
    Describe((uintptr_t)ptr, 1);
    ShowStatsAndAbort();
  }
  return usable_size;
}

size_t asan_mz_size(const void *ptr) {
  return malloc_info.AllocationSize((uintptr_t)ptr);
}

void DescribeHeapAddress(uintptr_t addr, uintptr_t access_size) {
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
  CHECK(REAL(memset) != NULL);
  REAL(memset)(this, 0, sizeof(*this));
}

bool FakeStack::AddrIsInSizeClass(uintptr_t addr, size_t size_class) {
  uintptr_t mem = allocated_size_classes_[size_class];
  uintptr_t size = ClassMmapSize(size_class);
  bool res = mem && addr >= mem && addr < mem + size;
  return res;
}

uintptr_t FakeStack::AddrIsInFakeStack(uintptr_t addr) {
  for (size_t i = 0; i < kNumberOfSizeClasses; i++) {
    if (AddrIsInSizeClass(addr, i)) return allocated_size_classes_[i];
  }
  return 0;
}

// We may want to compute this during compilation.
inline size_t FakeStack::ComputeSizeClass(size_t alloc_size) {
  size_t rounded_size = RoundUpToPowerOfTwo(alloc_size);
  size_t log = Log2(rounded_size);
  CHECK(alloc_size <= (1UL << log));
  if (!(alloc_size > (1UL << (log-1)))) {
    Printf("alloc_size %zu log %zu\n", alloc_size, log);
  }
  CHECK(alloc_size > (1UL << (log-1)));
  size_t res = log < kMinStackFrameSizeLog ? 0 : log - kMinStackFrameSizeLog;
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

void FakeStack::Init(size_t stack_size) {
  stack_size_ = stack_size;
  alive_ = true;
}

void FakeStack::Cleanup() {
  alive_ = false;
  for (size_t i = 0; i < kNumberOfSizeClasses; i++) {
    uintptr_t mem = allocated_size_classes_[i];
    if (mem) {
      PoisonShadow(mem, ClassMmapSize(i), 0);
      allocated_size_classes_[i] = 0;
      AsanUnmapOrDie((void*)mem, ClassMmapSize(i));
    }
  }
}

size_t FakeStack::ClassMmapSize(size_t size_class) {
  return RoundUpToPowerOfTwo(stack_size_);
}

void FakeStack::AllocateOneSizeClass(size_t size_class) {
  CHECK(ClassMmapSize(size_class) >= kPageSize);
  uintptr_t new_mem = (uintptr_t)AsanMmapSomewhereOrDie(
      ClassMmapSize(size_class), __FUNCTION__);
  // Printf("T%d new_mem[%zu]: %p-%p mmap %zu\n",
  //       asanThreadRegistry().GetCurrent()->tid(),
  //       size_class, new_mem, new_mem + ClassMmapSize(size_class),
  //       ClassMmapSize(size_class));
  size_t i;
  for (i = 0; i < ClassMmapSize(size_class);
       i += ClassSize(size_class)) {
    size_classes_[size_class].FifoPush((FakeFrame*)(new_mem + i));
  }
  CHECK(i == ClassMmapSize(size_class));
  allocated_size_classes_[size_class] = new_mem;
}

uintptr_t FakeStack::AllocateStack(size_t size, size_t real_stack) {
  if (!alive_) return real_stack;
  CHECK(size <= kMaxStackMallocSize && size > 1);
  size_t size_class = ComputeSizeClass(size);
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
  uintptr_t ptr = (uintptr_t)fake_frame;
  PoisonShadow(ptr, size, 0);
  return ptr;
}

void FakeStack::DeallocateFrame(FakeFrame *fake_frame) {
  CHECK(alive_);
  size_t size = fake_frame->size_minus_one + 1;
  size_t size_class = ComputeSizeClass(size);
  CHECK(allocated_size_classes_[size_class]);
  uintptr_t ptr = (uintptr_t)fake_frame;
  CHECK(AddrIsInSizeClass(ptr, size_class));
  CHECK(AddrIsInSizeClass(ptr + size - 1, size_class));
  size_classes_[size_class].FifoPush(fake_frame);
}

void FakeStack::OnFree(size_t ptr, size_t size, size_t real_stack) {
  FakeFrame *fake_frame = (FakeFrame*)ptr;
  CHECK(fake_frame->magic = kRetiredStackFrameMagic);
  CHECK(fake_frame->descr != 0);
  CHECK(fake_frame->size_minus_one == size - 1);
  PoisonShadow(ptr, size, kAsanStackAfterReturnMagic);
}

}  // namespace __asan

// ---------------------- Interface ---------------- {{{1
using namespace __asan;  // NOLINT

size_t __asan_stack_malloc(size_t size, size_t real_stack) {
  if (!FLAG_use_fake_stack) return real_stack;
  AsanThread *t = asanThreadRegistry().GetCurrent();
  if (!t) {
    // TSD is gone, use the real stack.
    return real_stack;
  }
  size_t ptr = t->fake_stack().AllocateStack(size, real_stack);
  // Printf("__asan_stack_malloc %p %zu %p\n", ptr, size, real_stack);
  return ptr;
}

void __asan_stack_free(size_t ptr, size_t size, size_t real_stack) {
  if (!FLAG_use_fake_stack) return;
  if (ptr != real_stack) {
    FakeStack::OnFree(ptr, size, real_stack);
  }
}

// ASan allocator doesn't reserve extra bytes, so normally we would
// just return "size".
size_t __asan_get_estimated_allocated_size(size_t size) {
  if (size == 0) return 1;
  return Min(size, kMaxAllowedMallocSize);
}

bool __asan_get_ownership(const void *p) {
  return malloc_info.AllocationSize((uintptr_t)p) > 0;
}

size_t __asan_get_allocated_size(const void *p) {
  if (p == NULL) return 0;
  size_t allocated_size = malloc_info.AllocationSize((uintptr_t)p);
  // Die if p is not malloced or if it is already freed.
  if (allocated_size == 0) {
    Report("ERROR: AddressSanitizer attempting to call "
           "__asan_get_allocated_size() for pointer which is "
           "not owned: %p\n", p);
    PRINT_CURRENT_STACK();
    Describe((uintptr_t)p, 1);
    ShowStatsAndAbort();
  }
  return allocated_size;
}
