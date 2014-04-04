//===-- msan_allocator.cc --------------------------- ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
// MemorySanitizer allocator.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "msan.h"
#include "msan_allocator.h"
#include "msan_thread.h"

namespace __msan {

struct Metadata {
  uptr requested_size;
};

struct MsanMapUnmapCallback {
  void OnMap(uptr p, uptr size) const {}
  void OnUnmap(uptr p, uptr size) const {
    __msan_unpoison((void *)p, size);

    // We are about to unmap a chunk of user memory.
    // Mark the corresponding shadow memory as not needed.
    FlushUnneededShadowMemory(MEM_TO_SHADOW(p), size);
    if (__msan_get_track_origins())
      FlushUnneededShadowMemory(MEM_TO_ORIGIN(p), size);
  }
};

static const uptr kAllocatorSpace = 0x600000000000ULL;
static const uptr kAllocatorSize   = 0x80000000000;  // 8T.
static const uptr kMetadataSize  = sizeof(Metadata);
static const uptr kMaxAllowedMallocSize = 8UL << 30;

typedef SizeClassAllocator64<kAllocatorSpace, kAllocatorSize, kMetadataSize,
                             DefaultSizeClassMap,
                             MsanMapUnmapCallback> PrimaryAllocator;
typedef SizeClassAllocatorLocalCache<PrimaryAllocator> AllocatorCache;
typedef LargeMmapAllocator<MsanMapUnmapCallback> SecondaryAllocator;
typedef CombinedAllocator<PrimaryAllocator, AllocatorCache,
                          SecondaryAllocator> Allocator;

static Allocator allocator;
static AllocatorCache fallback_allocator_cache;
static SpinMutex fallback_mutex;

static int inited = 0;

static inline void Init() {
  if (inited) return;
  __msan_init();
  inited = true;  // this must happen before any threads are created.
  allocator.Init();
}

AllocatorCache *GetAllocatorCache(MsanThreadLocalMallocStorage *ms) {
  CHECK(ms);
  CHECK_LE(sizeof(AllocatorCache), sizeof(ms->allocator_cache));
  return reinterpret_cast<AllocatorCache *>(ms->allocator_cache);
}

void MsanThreadLocalMallocStorage::CommitBack() {
  allocator.SwallowCache(GetAllocatorCache(this));
}

static void *MsanAllocate(StackTrace *stack, uptr size, uptr alignment,
                          bool zeroise) {
  Init();
  if (size > kMaxAllowedMallocSize) {
    Report("WARNING: MemorySanitizer failed to allocate %p bytes\n",
           (void *)size);
    return AllocatorReturnNull();
  }
  MsanThread *t = GetCurrentThread();
  void *allocated;
  if (t) {
    AllocatorCache *cache = GetAllocatorCache(&t->malloc_storage());
    allocated = allocator.Allocate(cache, size, alignment, false);
  } else {
    SpinMutexLock l(&fallback_mutex);
    AllocatorCache *cache = &fallback_allocator_cache;
    allocated = allocator.Allocate(cache, size, alignment, false);
  }
  Metadata *meta =
      reinterpret_cast<Metadata *>(allocator.GetMetaData(allocated));
  meta->requested_size = size;
  if (zeroise) {
    __msan_clear_and_unpoison(allocated, size);
  } else if (flags()->poison_in_malloc) {
    __msan_poison(allocated, size);
    if (__msan_get_track_origins()) {
      u32 stack_id = StackDepotPut(stack->trace, stack->size);
      CHECK(stack_id);
      CHECK_EQ((stack_id >> 31),
               0);  // Higher bit is occupied by stack origins.
      __msan_set_origin(allocated, size, stack_id);
    }
  }
  MSAN_MALLOC_HOOK(allocated, size);
  return allocated;
}

void MsanDeallocate(StackTrace *stack, void *p) {
  CHECK(p);
  Init();
  MSAN_FREE_HOOK(p);
  Metadata *meta = reinterpret_cast<Metadata*>(allocator.GetMetaData(p));
  uptr size = meta->requested_size;
  meta->requested_size = 0;
  // This memory will not be reused by anyone else, so we are free to keep it
  // poisoned.
  if (flags()->poison_in_free) {
    __msan_poison(p, size);
    if (__msan_get_track_origins()) {
      u32 stack_id = StackDepotPut(stack->trace, stack->size);
      CHECK(stack_id);
      CHECK_EQ((stack_id >> 31),
               0);  // Higher bit is occupied by stack origins.
      __msan_set_origin(p, size, stack_id);
    }
  }
  MsanThread *t = GetCurrentThread();
  if (t) {
    AllocatorCache *cache = GetAllocatorCache(&t->malloc_storage());
    allocator.Deallocate(cache, p);
  } else {
    SpinMutexLock l(&fallback_mutex);
    AllocatorCache *cache = &fallback_allocator_cache;
    allocator.Deallocate(cache, p);
  }
}

void *MsanReallocate(StackTrace *stack, void *old_p, uptr new_size,
                     uptr alignment, bool zeroise) {
  if (!old_p)
    return MsanAllocate(stack, new_size, alignment, zeroise);
  if (!new_size) {
    MsanDeallocate(stack, old_p);
    return 0;
  }
  Metadata *meta = reinterpret_cast<Metadata*>(allocator.GetMetaData(old_p));
  uptr old_size = meta->requested_size;
  uptr actually_allocated_size = allocator.GetActuallyAllocatedSize(old_p);
  if (new_size <= actually_allocated_size) {
    // We are not reallocating here.
    meta->requested_size = new_size;
    if (new_size > old_size)
      __msan_poison((char*)old_p + old_size, new_size - old_size);
    return old_p;
  }
  uptr memcpy_size = Min(new_size, old_size);
  void *new_p = MsanAllocate(stack, new_size, alignment, zeroise);
  // Printf("realloc: old_size %zd new_size %zd\n", old_size, new_size);
  if (new_p) {
    __msan_memcpy(new_p, old_p, memcpy_size);
    MsanDeallocate(stack, old_p);
  }
  return new_p;
}

static uptr AllocationSize(const void *p) {
  if (p == 0)
    return 0;
  const void *beg = allocator.GetBlockBegin(p);
  if (beg != p)
    return 0;
  Metadata *b = (Metadata*)allocator.GetMetaData(p);
  return b->requested_size;
}

}  // namespace __msan

using namespace __msan;

uptr __msan_get_current_allocated_bytes() {
  u64 stats[AllocatorStatCount];
  allocator.GetStats(stats);
  u64 m = stats[AllocatorStatMalloced];
  u64 f = stats[AllocatorStatFreed];
  return m >= f ? m - f : 1;
}

uptr __msan_get_heap_size() {
  u64 stats[AllocatorStatCount];
  allocator.GetStats(stats);
  u64 m = stats[AllocatorStatMmapped];
  u64 f = stats[AllocatorStatUnmapped];
  return m >= f ? m - f : 1;
}

uptr __msan_get_free_bytes() {
  return 1;
}

uptr __msan_get_unmapped_bytes() {
  return 1;
}

uptr __msan_get_estimated_allocated_size(uptr size) {
  return size;
}

int __msan_get_ownership(const void *p) {
  return AllocationSize(p) != 0;
}

uptr __msan_get_allocated_size(const void *p) {
  return AllocationSize(p);
}
