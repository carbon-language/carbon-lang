//===-- sanitizer_deadlock_detector.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of Sanitizer runtime.
// The deadlock detector maintains a directed graph of lock acquisitions.
// When a lock event happens, the detector checks if the locks already held by
// the current thread are reachable from the newly acquired lock.
//
// FIXME: this is work in progress, nothing really works yet.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_DEADLOCK_DETECTOR_H
#define SANITIZER_DEADLOCK_DETECTOR_H

#include "sanitizer_common.h"
#include "sanitizer_bvgraph.h"

namespace __sanitizer {

// Thread-local state for DeadlockDetector.
// It contains the locks currently held by the owning thread.
template <class BV>
class DeadlockDetectorTLS {
 public:
  // No CTOR.
  void clear() {
    bv_.clear();
    epoch_ = 0;
  }

  void addLock(uptr lock_id, uptr current_epoch) {
    if (current_epoch != epoch_)  {
      bv_.clear();
      epoch_ = current_epoch;
    }
    CHECK(bv_.setBit(lock_id));
  }

  void removeLock(uptr lock_id, uptr current_epoch) {
    if (current_epoch != epoch_)  {
      bv_.clear();
      epoch_ = current_epoch;
    }
    CHECK(bv_.clearBit(lock_id));
  }

  const BV &getLocks() const { return bv_; }

 private:
  BV bv_;
  uptr epoch_;
};

// DeadlockDetector.
// For deadlock detection to work we need one global DeadlockDetector object
// and one DeadlockDetectorTLS object per evey thread.
// This class is not thread safe, all concurrent accesses should be guarded
// by an external lock.
// Not thread-safe, all accesses should be protected by an external lock.
template <class BV>
class DeadlockDetector {
 public:
  typedef BV BitVector;

  uptr size() const { return g_.size(); }

  // No CTOR.
  void clear() {
    current_epoch_ = 0;
    available_nodes_.clear();
    recycled_nodes_.clear();
    g_.clear();
  }

  // Allocate new deadlock detector node.
  // If we are out of available nodes first try to recycle some.
  // If there is nothing to recycle, flush the graph and increment the epoch.
  // Associate 'data' (opaque user's object) with the new node.
  uptr newNode(uptr data) {
    if (!available_nodes_.empty())
      return getAvailableNode(data);
    if (!recycled_nodes_.empty()) {
      CHECK(available_nodes_.empty());
      // removeEdgesFrom was called in removeNode.
      g_.removeEdgesTo(recycled_nodes_);
      available_nodes_.setUnion(recycled_nodes_);
      recycled_nodes_.clear();
      return getAvailableNode(data);
    }
    // We are out of vacant nodes. Flush and increment the current_epoch_.
    current_epoch_ += size();
    recycled_nodes_.clear();
    available_nodes_.setAll();
    g_.clear();
    return getAvailableNode(data);
  }

  // Get data associated with the node created by newNode().
  uptr getData(uptr node) const { return data_[nodeToIndex(node)]; }

  void removeNode(uptr node) {
    uptr idx = nodeToIndex(node);
    CHECK(!available_nodes_.getBit(idx));
    CHECK(recycled_nodes_.setBit(idx));
    g_.removeEdgesFrom(idx);
  }

  // Handle the lock event, return true if there is a cycle.
  // FIXME: handle RW locks, recusive locks, etc.
  bool onLock(DeadlockDetectorTLS<BV> *dtls, uptr cur_node) {
    uptr cur_idx = nodeToIndex(cur_node);
    bool is_reachable = g_.isReachable(cur_idx, dtls->getLocks());
    dtls->addLock(cur_idx, current_epoch_);
    g_.addEdges(dtls->getLocks(), cur_idx);
    return is_reachable;
  }

  // Handle the unlock event.
  void onUnlock(DeadlockDetectorTLS<BV> *dtls, uptr node) {
    dtls->removeLock(nodeToIndex(node), current_epoch_);
  }

  uptr testOnlyGetEpoch() const { return current_epoch_; }

 private:
  void check_idx(uptr idx) const { CHECK_LT(idx, size()); }

  void check_node(uptr node) const {
    CHECK_GE(node, size());
    CHECK_EQ(current_epoch_, node / size() * size());
  }

  uptr indexToNode(uptr idx) const {
    check_idx(idx);
    return idx + current_epoch_;
  }

  uptr nodeToIndex(uptr node) const {
    check_node(node);
    return node % size();
  }

  uptr getAvailableNode(uptr data) {
    uptr idx = available_nodes_.getAndClearFirstOne();
    data_[idx] = data;
    return indexToNode(idx);
  }

  uptr current_epoch_;
  BV available_nodes_;
  BV recycled_nodes_;
  BVGraph<BV> g_;
  uptr data_[BV::kSize];
};

} // namespace __sanitizer

#endif // SANITIZER_DEADLOCK_DETECTOR_H
