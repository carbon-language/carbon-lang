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
#include "sanitizer_mutex.h"

namespace __sanitizer {

struct StackDesc {
  StackDesc *link;
  u32 id;
  uptr hash;
  uptr size;
  uptr stack[1];
};

static struct {
  StaticSpinMutex mtx;
  StackDesc *head;
  u8 *region_pos;
  u8 *region_end;
  u32 seq;
} depot;

static uptr hash(const uptr *stack, uptr size) {
  return 0;
}

static StackDesc *allocDesc(uptr size) {
  uptr memsz = sizeof(StackDesc) + (size - 1) * sizeof(uptr);
  if (depot.region_pos + memsz > depot.region_end) {
    uptr allocsz = 64*1024;
    if (allocsz < memsz)
      allocsz = memsz;
    depot.region_pos = (u8*)MmapOrDie(allocsz, "stack depot");
    depot.region_end = depot.region_pos + allocsz;
  }
  StackDesc *s = (StackDesc*)depot.region_pos;
  depot.region_pos += memsz;
  return s;
}

u32 StackDepotPut(const uptr *stack, uptr size) {
  if (stack == 0 || size == 0)
    return 0;
  uptr h = hash(stack, size);
  SpinMutexLock l(&depot.mtx);
  for (StackDesc *s = depot.head; s; s = s->link) {
    if (s->hash == h && s->size == size
        && internal_memcmp(s->stack, stack, size * sizeof(uptr)) == 0)
      return s->id;
  }
  StackDesc *s = allocDesc(size);
  s->id = ++depot.seq;
  s->hash = h;
  s->size = size;
  internal_memcpy(s->stack, stack, size * sizeof(uptr));
  s->link = depot.head;
  depot.head = s;
  return s->id;
}

const uptr *StackDepotGet(u32 id, uptr *size) {
  if (id == 0)
    return 0;
  SpinMutexLock l(&depot.mtx);
  for (StackDesc *s = depot.head; s; s = s->link) {
    if (s->id == id) {
      *size = s->size;
      return s->stack;
    }
  }
  return 0;
}

}  // namespace __sanitizer
