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
class DeadlockDetectorTLS {
 public:
  // No CTOR.
  void clear() { n_locks_ = 0; }
  void addLock(uptr node) {
    CHECK_LT(n_locks_, ARRAY_SIZE(locks_));
    locks_[n_locks_++] = node;
  }
  void removeLock(uptr node) {
    CHECK_NE(n_locks_, 0U);
    for (sptr i = n_locks_ - 1; i >= 0; i--) {
      if (locks_[i] == node) {
        locks_[i] = locks_[n_locks_ - 1];
        n_locks_--;
        return;
      }
    }
    CHECK(0);
  }
  uptr numLocks() const { return n_locks_; }
  uptr getLock(uptr idx) const {
    CHECK_LT(idx, n_locks_);
    return locks_[idx];
  }

 private:
  uptr n_locks_;
  uptr locks_[64];
};

// DeadlockDetector.
// For deadlock detection to work we need one global DeadlockDetector object
// and one DeadlockDetectorTLS object per evey thread.
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
      available_nodes_.setUnion(recycled_nodes_);
      recycled_nodes_.clear();
      // FIXME: actually recycle nodes in the graph.
      return getAvailableNode(data);
    }
    // We are out of vacant nodes. Flush and increment the current_epoch_.
    uptr new_epoch = current_epoch_ + BV::kSize;
    clear();
    current_epoch_ = new_epoch;
    available_nodes_.setAll();
    return getAvailableNode(data);
  }

  // Get data associated with the node created by newNode().
  uptr getData(uptr node) const { return data_[nodeToIndex(node)]; }

  void removeNode(uptr node) {
    uptr idx = nodeToIndex(node);
    CHECK(!available_nodes_.getBit(idx));
    CHECK(recycled_nodes_.setBit(idx));
    // FIXME: also remove from the graph.
  }

  // Handle the lock event, return true if there is a cycle.
  // FIXME: handle RW locks, recusive locks, etc.
  bool onLock(DeadlockDetectorTLS *dtls, uptr cur_node) {
    BV cur_locks;
    cur_locks.clear();
    uptr cur_idx = nodeToIndex(cur_node);
    for (uptr i = 0, n = dtls->numLocks(); i < n; i++) {
      uptr prev_node = dtls->getLock(i);
      uptr prev_idx = nodeToIndex(prev_node);
      g_.addEdge(prev_idx, cur_idx);
      cur_locks.setBit(prev_idx);
      // Printf("OnLock %zx; prev %zx\n", cur_node, dtls->getLock(i));
    }
    dtls->addLock(cur_node);
    return g_.isReachable(cur_idx, cur_locks);
  }

  // Handle the unlock event.
  void onUnlock(DeadlockDetectorTLS *dtls, uptr node) {
    dtls->removeLock(node);
  }

 private:
  void check_idx(uptr idx) const { CHECK_LT(idx, size()); }
  void check_node(uptr node) const {
    CHECK_GE(node, size());
    CHECK_EQ(current_epoch_, node / size() * size());
  }
  uptr indexToNode(uptr idx) {
    check_idx(idx);
    return idx | current_epoch_;
  }
  uptr nodeToIndex(uptr node) {
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
