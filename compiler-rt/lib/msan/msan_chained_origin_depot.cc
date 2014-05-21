//===-- msan_chained_origin_depot.cc -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A storage for chained origins.
//===----------------------------------------------------------------------===//

#include "msan_chained_origin_depot.h"

#include "sanitizer_common/sanitizer_stackdepotbase.h"

namespace __msan {

struct ChainedOriginDepotDesc {
  u32 here_id;
  u32 prev_id;
  u32 hash() const { return here_id ^ prev_id; }
  bool is_valid() { return true; }
};

struct ChainedOriginDepotNode {
  ChainedOriginDepotNode *link;
  u32 id;
  u32 here_id;
  u32 prev_id;

  typedef ChainedOriginDepotDesc args_type;
  bool eq(u32 hash, const args_type &args) const {
    return here_id == args.here_id && prev_id == args.prev_id;
  }
  static uptr storage_size(const args_type &args) {
    return sizeof(ChainedOriginDepotNode);
  }
  void store(const args_type &args, u32 other_hash) {
    here_id = args.here_id;
    prev_id = args.prev_id;
  }
  args_type load() const {
    args_type ret = {here_id, prev_id};
    return ret;
  }
  struct Handle {
    ChainedOriginDepotNode *node_;
    Handle() : node_(0) {}
    explicit Handle(ChainedOriginDepotNode *node) : node_(node) {}
    bool valid() { return node_; }
    u32 id() { return node_->id; }
    int here_id() { return node_->here_id; }
    int prev_id() { return node_->prev_id; }
  };
  Handle get_handle() { return Handle(this); }

  typedef Handle handle_type;
};

static StackDepotBase<ChainedOriginDepotNode, 3> chainedOriginDepot;

StackDepotStats *ChainedOriginDepotGetStats() {
  return chainedOriginDepot.GetStats();
}

bool ChainedOriginDepotPut(u32 here_id, u32 prev_id, u32 *new_id) {
  ChainedOriginDepotDesc desc = {here_id, prev_id};
  bool inserted;
  ChainedOriginDepotNode::Handle h = chainedOriginDepot.Put(desc, &inserted);
  *new_id = h.valid() ? h.id() : 0;
  return inserted;
}

// Retrieves a stored stack trace by the id.
u32 ChainedOriginDepotGet(u32 id, u32 *other) {
  ChainedOriginDepotDesc desc = chainedOriginDepot.Get(id);
  *other = desc.prev_id;
  return desc.here_id;
}

}  // namespace __msan
