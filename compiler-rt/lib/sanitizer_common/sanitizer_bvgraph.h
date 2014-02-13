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
    check(from|to);
    return v[from].setBit(to);
  }

  bool hasEdge(uptr from, uptr to) const {
    check(from|to);
    return v[from].getBit(to);
  }

  // Returns true if there is a path from the node 'from'
  // to any of the nodes in 'targets'.
  bool isReachable(uptr from, const BV &targets) {
    BV to_visit, visited;
    to_visit.clear();
    to_visit.setUnion(v[from]);
    visited.clear();
    visited.setBit(from);
    while (!to_visit.empty()) {
      uptr idx = to_visit.getAndClearFirstOne();
      if (visited.setBit(idx))
        to_visit.setUnion(v[idx]);
    }
    return targets.intersectsWith(visited);
  }

 private:
  void check(uptr idx) const { CHECK_LE(idx, size()); }
  BV v[kSize];
};

} // namespace __sanitizer

#endif // SANITIZER_BVGRAPH_H
