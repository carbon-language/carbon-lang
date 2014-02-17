//===-- sanitizer_bvgraph.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of Sanitizer runtime.
// BVGraph -- a directed graph.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_BVGRAPH_H
#define SANITIZER_BVGRAPH_H

#include "sanitizer_common.h"
#include "sanitizer_bitvector.h"

namespace __sanitizer {

// Directed graph of fixed size implemented as an array of bit vectors.
// Not thread-safe, all accesses should be protected by an external lock.
template<class BV>
class BVGraph {
 public:
  enum SizeEnum { kSize = BV::kSize };
  uptr size() const { return kSize; }
  // No CTOR.
  void clear() {
    for (uptr i = 0; i < size(); i++)
      v[i].clear();
  }

  // Returns true if a new edge was added.
  bool addEdge(uptr from, uptr to) {
    check(from, to);
    return v[from].setBit(to);
  }

  // Returns true if at least one new edge was added.
  bool addEdges(const BV &from, uptr to) {
    bool res = false;
    t1.copyFrom(from);
    while (!t1.empty())
      if (v[t1.getAndClearFirstOne()].setBit(to))
        res = true;
    return res;
  }

  bool hasEdge(uptr from, uptr to) const {
    check(from, to);
    return v[from].getBit(to);
  }

  // Returns true if there is a path from the node 'from'
  // to any of the nodes in 'targets'.
  bool isReachable(uptr from, const BV &targets) {
    BV &to_visit = t1,
       &visited = t2;
    to_visit.copyFrom(v[from]);
    visited.clear();
    visited.setBit(from);
    while (!to_visit.empty()) {
      uptr idx = to_visit.getAndClearFirstOne();
      if (visited.setBit(idx))
        to_visit.setUnion(v[idx]);
    }
    return targets.intersectsWith(visited);
  }

  // Finds a path from 'from' to one of the nodes in 'target',
  // stores up to 'path_size' items of the path into 'path',
  // returns the path length, or 0 if there is no path of size 'path_size'.
  uptr findPath(uptr from, const BV &targets, uptr *path, uptr path_size) {
    if (path_size == 0)
      return 0;
    path[0] = from;
    if (targets.getBit(from))
      return 1;
    // The function is recursive, so we don't want to create BV on stack.
    // Instead of a getAndClearFirstOne loop we use the slower iterator.
    for (typename BV::Iterator it(v[from]); it.hasNext(); ) {
      uptr idx = it.next();
      if (uptr res = findPath(idx, targets, path + 1, path_size - 1))
        return res + 1;
    }
    return 0;
  }

 private:
  void check(uptr idx1, uptr idx2) const {
    CHECK_LT(idx1, size());
    CHECK_LT(idx2, size());
  }
  BV v[kSize];
  // Keep temporary vectors here since we can not create large objects on stack.
  BV t1, t2;
};

} // namespace __sanitizer

#endif // SANITIZER_BVGRAPH_H
