//===-- msan_origin.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Origin id utils.
//===----------------------------------------------------------------------===//
#ifndef MSAN_ORIGIN_H
#define MSAN_ORIGIN_H

namespace __msan {

// Origin handling.
//
// Origin is a 32-bit identifier that is attached to any uninitialized value in
// the program and describes, more or less exactly, how this memory came to be
// uninitialized.
//
// Origin ids are values of ChainedOriginDepot, which is a mapping of (stack_id,
// prev_id) -> id, where
//  * stack_id describes an event in the program, usually a memory store.
//    StackDepot keeps a mapping between those and corresponding stack traces.
//  * prev_id is another origin id that describes the earlier part of the
//    uninitialized value history.
// Following a chain of prev_id provides the full recorded history of an
// uninitialized value.
//
// This, effectively, defines a tree (or 2 trees, see below) where nodes are
// points in value history marked with origin ids, and edges are events that are
// marked with stack_id.
//
// There are 2 special root origin ids:
// * kHeapRoot - an origin with prev_id == kHeapRoot describes an event of
//   allocating memory from heap.
// * kStackRoot - an origin with prev_id == kStackRoot describes an event of
//   allocating memory from stack (i.e. on function entry).
// Note that ChainedOriginDepot does not store any node for kHeapRoot or
// kStackRoot. These are just special id values.
//
// Three highest bits of origin id are used to store the length (or depth) of
// the origin chain. Special depth value of 0 means unlimited.

class Origin {
 public:
  static const int kDepthBits = 3;
  static const int kDepthShift = 32 - kDepthBits;
  static const u32 kIdMask = ((u32)-1) >> (32 - kDepthShift);
  static const u32 kDepthMask = ~kIdMask;

  static const int kMaxDepth = (1 << kDepthBits) - 1;

  static const u32 kHeapRoot = (u32)-1;
  static const u32 kStackRoot = (u32)-2;

  explicit Origin(u32 raw_id) : raw_id_(raw_id) {}
  Origin(u32 id, u32 depth) : raw_id_((depth << kDepthShift) | id) {
    CHECK_EQ(this->depth(), depth);
    CHECK_EQ(this->id(), id);
  }
  int depth() const { return raw_id_ >> kDepthShift; }
  u32 id() const { return raw_id_ & kIdMask; }
  u32 raw_id() const { return raw_id_; }
  bool isStackRoot() const { return raw_id_ == kStackRoot; }
  bool isHeapRoot() const { return raw_id_ == kHeapRoot; }

 private:
  u32 raw_id_;
};

}  // namespace __msan

#endif  // MSAN_ORIGIN_H
