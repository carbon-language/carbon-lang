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
#include "asan_thread.h"
#include "asan_thread_registry.h"
#include "sanitizer/asan_interface.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __asan {

struct AsanMapUnmapCallback {
  void OnMap(uptr p, uptr size) const {
    PoisonShadow(p, size, kAsanHeapLeftRedzoneMagic);
  }
  void OnUnmap(uptr p, uptr size) const {
    PoisonShadow(p, size, 0);
  }
};

#if SANITIZER_WORDSIZE == 64
const uptr kAllocatorSpace = 0x600000000000ULL;
const uptr kAllocatorSize  =  0x10000000000ULL;  // 1T.
typedef SizeClassAllocator64<kAllocatorSpace, kAllocatorSize, 0 /*metadata*/,
    DefaultSizeClassMap, AsanMapUnmapCallback> PrimaryAllocator;
#elif SANITIZER_WORDSIZE == 32
static const u64 kAddressSpaceSize = 1ULL << 32;
typedef SizeClassAllocator32<0, kAddressSpaceSize, 16,
  CompactSizeClassMap, AsanMapUnmapCallback> PrimaryAllocator;
#endif

typedef SizeClassAllocatorLocalCache<PrimaryAllocator> AllocatorCache;
typedef LargeMmapAllocator<AsanMapUnmapCallback> SecondaryAllocator;
typedef CombinedAllocator<PrimaryAllocator, AllocatorCache,
    SecondaryAllocator> Allocator;

static THREADLOCAL AllocatorCache cache;
static Allocator allocator;

static const uptr kMaxAllowedMallocSize =
    (SANITIZER_WORDSIZE == 32) ? 3UL << 30 : 8UL << 30;

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
//   H -- ChunkHeader (16 bytes on 64-bit arch, 8 bytes on 32-bit arch).
//     ChunkHeader is also a part of the left redzone.
//   U -- user memory.
//   R -- right redzone (0 or more bytes)
// ChunkBase consists of ChunkHeader and other bytes that overlap with user
// memory.

#if SANITIZER_WORDSIZE == 64
struct ChunkBase {
  // 1-st 8 bytes.
  uptr chunk_state       : 8;  // Must be first.
  uptr alloc_tid         : 24;
  uptr free_tid          : 24;
  uptr from_memalign     : 1;
  // 2-nd 8 bytes
  uptr user_requested_size;
  // End of ChunkHeader.
  // 3-rd 8 bytes. These overlap with the user memory.
  AsanChunk *next;
};

static const uptr kChunkHeaderSize = 16;
COMPILER_CHECK(sizeof(ChunkBase) == 24);

#elif SANITIZER_WORDSIZE == 32
struct ChunkBase {
  // 1-st 8 bytes.
  uptr chunk_state       : 8;  // Must be first.
  uptr from_memalign     : 1;
  uptr alloc_tid         : 23;
  uptr user_requested_size;
  // End of ChunkHeader.
  // 2-nd 8 bytes. These overlap with the user memory.
  AsanChunk *next;
  uptr  free_tid;
};

COMPILER_CHECK(sizeof(ChunkBase) == 16);
static const uptr kChunkHeaderSize = 8;
#endif

static uptr ComputeRZSize(uptr user_requested_size) {
  // FIXME: implement adaptive redzones.
  return flags()->redzone;
}

struct AsanChunk: ChunkBase {
  uptr Beg() { return reinterpret_cast<uptr>(this) + kChunkHeaderSize; }
  uptr UsedSize() { return user_requested_size; }
  // We store the alloc/free stack traces in the chunk itself.
  uptr AllocStackBeg() {
    return Beg() - ComputeRZSize(user_requested_size);
  }
  uptr AllocStackSize() {
    return ComputeRZSize(user_requested_size) - kChunkHeaderSize;
  }
  uptr FreeStackBeg();
  uptr FreeStackSize();
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
  stack->size = 0;
}

static const uptr kReturnOnZeroMalloc = 0x0123;  // Zero page is protected.

static void *Allocate(uptr size, uptr alignment, StackTrace *stack) {
  Init();
  CHECK(stack);
  if (alignment < 8) alignment = 8;
  if (size == 0)
    return reinterpret_cast<void *>(kReturnOnZeroMalloc);
  CHECK(IsPowerOfTwo(alignment));
  uptr rz_size = ComputeRZSize(size);
  uptr rounded_size = RoundUpTo(size, rz_size);
  uptr needed_size = rounded_size + rz_size;
  if (alignment > rz_size)
    needed_size += alignment;
  CHECK(IsAligned(needed_size, rz_size));
  if (size > kMaxAllowedMallocSize || needed_size > kMaxAllowedMallocSize) {
    Report("WARNING: AddressSanitizer failed to allocate %p bytes\n",
           (void*)size);
    return 0;
  }

  AsanThread *t = asanThreadRegistry().GetCurrent();
  void *allocated = allocator.Allocate(&cache, needed_size, 8, false);
  uptr alloc_beg = reinterpret_cast<uptr>(allocated);
  uptr alloc_end = alloc_beg + needed_size;
  uptr beg_plus_redzone = alloc_beg + rz_size;
  uptr user_beg = beg_plus_redzone;
  if (!IsAligned(user_beg, alignment))
    user_beg = RoundUpTo(user_beg, alignment);
  uptr user_end = user_beg + size;
  CHECK_LE(user_end, alloc_end);
  uptr chunk_beg = user_beg - kChunkHeaderSize;
//  Printf("allocated: %p beg_plus_redzone %p chunk_beg %p\n",
//         allocated, beg_plus_redzone, chunk_beg);
  AsanChunk *m = reinterpret_cast<AsanChunk *>(chunk_beg);
  m->chunk_state = CHUNK_ALLOCATED;
  u32 alloc_tid = t ? t->tid() : 0;
  m->alloc_tid = alloc_tid;
  CHECK_EQ(alloc_tid, m->alloc_tid);  // Does alloc_tid fit into the bitfield?
  m->free_tid = kInvalidTid;
  m->from_memalign = user_beg != beg_plus_redzone;
  m->user_requested_size = size;
  StackTrace::CompressStack(stack, m->AllocStackBeg(), m->AllocStackSize());

  uptr size_rounded_down_to_granularity = RoundDownTo(size, SHADOW_GRANULARITY);
  // Unpoison the bulk of the memory region.
  if (size_rounded_down_to_granularity)
    PoisonShadow(user_beg, size_rounded_down_to_granularity, 0);
  // Deal with the end of the region if size is not aligned to granularity.
  if (size != size_rounded_down_to_granularity) {
    u8 *shadow = (u8*)MemToShadow(user_beg + size_rounded_down_to_granularity);
    *shadow = size & (SHADOW_GRANULARITY - 1);
  }

  void *res = reinterpret_cast<void *>(user_beg);
  ASAN_MALLOC_HOOK(res, size);
  return res;
}

static void Deallocate(void *ptr, StackTrace *stack) {
  uptr p = reinterpret_cast<uptr>(ptr);
  if (p == 0 || p == kReturnOnZeroMalloc) return;
  uptr chunk_beg = p - kChunkHeaderSize;
  AsanChunk *m = reinterpret_cast<AsanChunk *>(chunk_beg);
  uptr alloc_beg = p - ComputeRZSize(m->user_requested_size);
  if (m->from_memalign)
    alloc_beg = reinterpret_cast<uptr>(allocator.GetBlockBegin(ptr));
  // Poison the region.
  PoisonShadow(m->Beg(), RoundUpTo(m->user_requested_size, SHADOW_GRANULARITY),
               kAsanHeapFreeMagic);
  ASAN_FREE_HOOK(ptr);
  allocator.Deallocate(&cache, reinterpret_cast<void *>(alloc_beg));
}

AsanChunkView FindHeapChunkByAddress(uptr address) {
  UNIMPLEMENTED();
  return AsanChunkView(0);
}

void AsanThreadLocalMallocStorage::CommitBack() {
  UNIMPLEMENTED();
}

SANITIZER_INTERFACE_ATTRIBUTE
void *asan_memalign(uptr alignment, uptr size, StackTrace *stack) {
  return Allocate(size, alignment, stack);
}

SANITIZER_INTERFACE_ATTRIBUTE
void asan_free(void *ptr, StackTrace *stack) {
  Deallocate(ptr, stack);
}

SANITIZER_INTERFACE_ATTRIBUTE
void *asan_malloc(uptr size, StackTrace *stack) {
  return Allocate(size, 8, stack);
}

void *asan_calloc(uptr nmemb, uptr size, StackTrace *stack) {
  void *ptr = Allocate(nmemb * size, 8, stack);
  if (ptr)
    REAL(memset)(ptr, 0, nmemb * size);
  return 0;
}

void *asan_realloc(void *p, uptr size, StackTrace *stack) {
  if (p == 0) {
    return Allocate(size, 8, stack);
  if (size == 0) {
    Deallocate(p, stack);
    return 0;
  }
  UNIMPLEMENTED;
  // return Reallocate((u8*)p, size, stack);
}

void *asan_valloc(uptr size, StackTrace *stack) {
  return Allocate(size, GetPageSizeCached(), stack);
}

void *asan_pvalloc(uptr size, StackTrace *stack) {
  uptr PageSize = GetPageSizeCached();
  size = RoundUpTo(size, PageSize);
  if (size == 0) {
    // pvalloc(0) should allocate one page.
    size = PageSize;
  }
  return Allocate(size, PageSize, stack);
}

int asan_posix_memalign(void **memptr, uptr alignment, uptr size,
                        StackTrace *stack) {
  void *ptr = Allocate(size, alignment, stack);
  CHECK(IsAligned((uptr)ptr, alignment));
  *memptr = ptr;
  return 0;
}

uptr asan_malloc_usable_size(void *ptr, StackTrace *stack) {
  UNIMPLEMENTED();
  return 0;
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
// just return "size".
uptr __asan_get_estimated_allocated_size(uptr size) {
  UNIMPLEMENTED();
  return 0;
}

bool __asan_get_ownership(const void *p) {
  UNIMPLEMENTED();
  return false;
}

uptr __asan_get_allocated_size(const void *p) {
  UNIMPLEMENTED();
  return 0;
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
