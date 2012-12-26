//===-- sanitizer_stackdepot.cc -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//

#include "sanitizer_stackdepot.h"
#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_mutex.h"
#include "sanitizer_atomic.h"

namespace __sanitizer {

const int kTabSize = 1024 * 1024;  // Hash table size.
const int kPartBits = 8;
const int kPartShift = sizeof(u32) * 8 - kPartBits - 1;
const int kPartCount = 1 << kPartBits;  // Number of subparts in the table.
const int kPartSize = kTabSize / kPartCount;
const int kMaxId = 1 << kPartShift;

struct StackDesc {
  StackDesc *link;
  u32 id;
  u32 hash;
  uptr size;
  uptr stack[1];  // [size]
};

static struct {
  StaticSpinMutex mtx;  // Protects alloc of new blocks for region allocator.
  atomic_uintptr_t region_pos;  // Region allocator for StackDesc's.
  atomic_uintptr_t region_end;
  atomic_uintptr_t tab[kTabSize];  // Hash table of StackDesc's.
  atomic_uint32_t seq[kPartCount];  // Unique id generators.
} depot;

static StackDepotStats stats;

StackDepotStats *StackDepotGetStats() {
  return &stats;
}

static u32 hash(const uptr *stack, uptr size) {
  // murmur2
  const u32 m = 0x5bd1e995;
  const u32 seed = 0x9747b28c;
  const u32 r = 24;
  u32 h = seed ^ (size * sizeof(uptr));
  for (uptr i = 0; i < size; i++) {
    u32 k = stack[i];
    k *= m;
    k ^= k >> r;
    k *= m;
    h *= m;
    h ^= k;
  }
  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;
  return h;
}

static StackDesc *tryallocDesc(uptr memsz) {
  // Optimisic lock-free allocation, essentially try to bump the region ptr.
  for (;;) {
    uptr cmp = atomic_load(&depot.region_pos, memory_order_acquire);
    uptr end = atomic_load(&depot.region_end, memory_order_acquire);
    if (cmp == 0 || cmp + memsz > end)
      return 0;
    if (atomic_compare_exchange_weak(
        &depot.region_pos, &cmp, cmp + memsz,
        memory_order_acquire))
      return (StackDesc*)cmp;
  }
}

static StackDesc *allocDesc(uptr size) {
  // First, try to allocate optimisitically.
  uptr memsz = sizeof(StackDesc) + (size - 1) * sizeof(uptr);
  StackDesc *s = tryallocDesc(memsz);
  if (s)
    return s;
  // If failed, lock, retry and alloc new superblock.
  SpinMutexLock l(&depot.mtx);
  for (;;) {
    s = tryallocDesc(memsz);
    if (s)
      return s;
    atomic_store(&depot.region_pos, 0, memory_order_relaxed);
    uptr allocsz = 64 * 1024;
    if (allocsz < memsz)
      allocsz = memsz;
    uptr mem = (uptr)MmapOrDie(allocsz, "stack depot");
    stats.mapped += allocsz;
    atomic_store(&depot.region_end, mem + allocsz, memory_order_release);
    atomic_store(&depot.region_pos, mem, memory_order_release);
  }
}

static u32 find(StackDesc *s, const uptr *stack, uptr size, u32 hash) {
  // Searches linked list s for the stack, returns its id.
  for (; s; s = s->link) {
    if (s->hash == hash && s->size == size) {
      uptr i = 0;
      for (; i < size; i++) {
        if (stack[i] != s->stack[i])
          break;
      }
      if (i == size)
        return s->id;
    }
  }
  return 0;
}

static StackDesc *lock(atomic_uintptr_t *p) {
  // Uses the pointer lsb as mutex.
  for (int i = 0;; i++) {
    uptr cmp = atomic_load(p, memory_order_relaxed);
    if ((cmp & 1) == 0
        && atomic_compare_exchange_weak(p, &cmp, cmp | 1,
                                        memory_order_acquire))
      return (StackDesc*)cmp;
    if (i < 10)
      proc_yield(10);
    else
      internal_sched_yield();
  }
}

static void unlock(atomic_uintptr_t *p, StackDesc *s) {
  DCHECK_EQ((uptr)s & 1, 0);
  atomic_store(p, (uptr)s, memory_order_release);
}

u32 StackDepotPut(const uptr *stack, uptr size) {
  if (stack == 0 || size == 0)
    return 0;
  uptr h = hash(stack, size);
  atomic_uintptr_t *p = &depot.tab[h % kTabSize];
  uptr v = atomic_load(p, memory_order_consume);
  StackDesc *s = (StackDesc*)(v & ~1);
  // First, try to find the existing stack.
  u32 id = find(s, stack, size, h);
  if (id)
    return id;
  // If failed, lock, retry and insert new.
  StackDesc *s2 = lock(p);
  if (s2 != s) {
    id = find(s2, stack, size, h);
    if (id) {
      unlock(p, s2);
      return id;
    }
  }
  uptr part = (h % kTabSize) / kPartSize;
  id = atomic_fetch_add(&depot.seq[part], 1, memory_order_relaxed) + 1;
  stats.n_uniq_ids++;
  CHECK_LT(id, kMaxId);
  id |= part << kPartShift;
  CHECK_NE(id, 0);
  CHECK_EQ(id & (1u << 31), 0);
  s = allocDesc(size);
  s->id = id;
  s->hash = h;
  s->size = size;
  internal_memcpy(s->stack, stack, size * sizeof(uptr));
  s->link = s2;
  unlock(p, s);
  return id;
}

const uptr *StackDepotGet(u32 id, uptr *size) {
  if (id == 0)
    return 0;
  CHECK_EQ(id & (1u << 31), 0);
  // High kPartBits contain part id, so we need to scan at most kPartSize lists.
  uptr part = id >> kPartShift;
  for (int i = 0; i != kPartSize; i++) {
    uptr idx = part * kPartSize + i;
    CHECK_LT(idx, kTabSize);
    atomic_uintptr_t *p = &depot.tab[idx];
    uptr v = atomic_load(p, memory_order_consume);
    StackDesc *s = (StackDesc*)(v & ~1);
    for (; s; s = s->link) {
      if (s->id == id) {
        *size = s->size;
        return s->stack;
      }
    }
  }
  *size = 0;
  return 0;
}

}  // namespace __sanitizer
