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
#include "sanitizer_stackdepotbase.h"

namespace __sanitizer {

struct StackDepotDesc {
  const uptr *stack;
  uptr size;
  u32 hash() const {
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
  bool is_valid() { return size > 0 && stack; }
};

struct StackDepotNode {
  StackDepotNode *link;
  u32 id;
  atomic_uint32_t hash_and_use_count; // hash_bits : 12; use_count : 20;
  uptr size;
  uptr stack[1];  // [size]

  static const u32 kUseCountBits = 20;
  static const u32 kMaxUseCount = 1 << kUseCountBits;
  static const u32 kUseCountMask = (1 << kUseCountBits) - 1;
  static const u32 kHashMask = ~kUseCountMask;

  typedef StackDepotDesc args_type;
  bool eq(u32 hash, const args_type &args) const {
    u32 hash_bits =
        atomic_load(&hash_and_use_count, memory_order_relaxed) & kHashMask;
    if ((hash & kHashMask) != hash_bits || args.size != size) return false;
    uptr i = 0;
    for (; i < size; i++) {
      if (stack[i] != args.stack[i]) return false;
    }
    return true;
  }
  static uptr storage_size(const args_type &args) {
    return sizeof(StackDepotNode) + (args.size - 1) * sizeof(uptr);
  }
  void store(const args_type &args, u32 hash) {
    atomic_store(&hash_and_use_count, hash & kHashMask, memory_order_relaxed);
    size = args.size;
    internal_memcpy(stack, args.stack, size * sizeof(uptr));
  }
  args_type load() const {
    args_type ret = {&stack[0], size};
    return ret;
  }
  StackDepotHandle get_handle() { return StackDepotHandle(this); }

  typedef StackDepotHandle handle_type;
};

COMPILER_CHECK(StackDepotNode::kMaxUseCount == (u32)kStackDepotMaxUseCount);

u32 StackDepotHandle::id() { return node_->id; }
int StackDepotHandle::use_count() {
  return atomic_load(&node_->hash_and_use_count, memory_order_relaxed) &
         StackDepotNode::kUseCountMask;
}
void StackDepotHandle::inc_use_count_unsafe() {
  u32 prev =
      atomic_fetch_add(&node_->hash_and_use_count, 1, memory_order_relaxed) &
      StackDepotNode::kUseCountMask;
  CHECK_LT(prev + 1, StackDepotNode::kMaxUseCount);
}
uptr StackDepotHandle::size() { return node_->size; }
uptr *StackDepotHandle::stack() { return &node_->stack[0]; }

// FIXME(dvyukov): this single reserved bit is used in TSan.
typedef StackDepotBase<StackDepotNode, 1> StackDepot;
static StackDepot theDepot;

StackDepotStats *StackDepotGetStats() {
  return theDepot.GetStats();
}

u32 StackDepotPut(const uptr *stack, uptr size) {
  StackDepotDesc desc = {stack, size};
  StackDepotHandle h = theDepot.Put(desc);
  return h.valid() ? h.id() : 0;
}

StackDepotHandle StackDepotPut_WithHandle(const uptr *stack, uptr size) {
  StackDepotDesc desc = {stack, size};
  return theDepot.Put(desc);
}

const uptr *StackDepotGet(u32 id, uptr *size) {
  StackDepotDesc desc = theDepot.Get(id);
  *size = desc.size;
  return desc.stack;
}

bool StackDepotReverseMap::IdDescPair::IdComparator(
    const StackDepotReverseMap::IdDescPair &a,
    const StackDepotReverseMap::IdDescPair &b) {
  return a.id < b.id;
}

StackDepotReverseMap::StackDepotReverseMap()
    : map_(StackDepotGetStats()->n_uniq_ids + 100) {
  for (int idx = 0; idx < StackDepot::kTabSize; idx++) {
    atomic_uintptr_t *p = &theDepot.tab[idx];
    uptr v = atomic_load(p, memory_order_consume);
    StackDepotNode *s = (StackDepotNode*)(v & ~1);
    for (; s; s = s->link) {
      IdDescPair pair = {s->id, s};
      map_.push_back(pair);
    }
  }
  InternalSort(&map_, map_.size(), IdDescPair::IdComparator);
}

const uptr *StackDepotReverseMap::Get(u32 id, uptr *size) {
  if (!map_.size()) return 0;
  IdDescPair pair = {id, 0};
  uptr idx = InternalBinarySearch(map_, 0, map_.size(), pair,
                                  IdDescPair::IdComparator);
  if (idx > map_.size()) {
    *size = 0;
    return 0;
  }
  StackDepotNode *desc = map_[idx].desc;
  *size = desc->size;
  return desc->stack;
}

}  // namespace __sanitizer
