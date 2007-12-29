//===- llvm/ADT/DenseSet.h - Dense probed hash table ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DenseSet class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_DENSESET_H
#define LLVM_ADT_DENSESET_H

#include "llvm/ADT/DenseMap.h"

namespace llvm {

/// DenseSet - This implements a dense probed hash-table based set.
///
/// FIXME: This is currently implemented directly in terms of DenseMap, this
/// should be optimized later if there is a need.
template<typename ValueT, typename ValueInfoT = DenseMapInfo<ValueT> >
class DenseSet {
  DenseMap<ValueT, char, ValueInfoT> TheMap;
public:
  DenseSet(const DenseSet &Other) : TheMap(Other.TheMap) {}
  explicit DenseSet(unsigned NumInitBuckets = 64) : TheMap(NumInitBuckets) {}
  
  bool empty() const { return TheMap.empty(); }
  unsigned size() const { return TheMap.size(); }
  
  // TODO add iterators.
  
  void clear() {
    TheMap.clear();
  }
  
  bool count(const ValueT &V) const {
    return TheMap.count(V);
  }
  
  void insert(const ValueT &V) {
    TheMap[V] = 0;
  }
  
  void erase(const ValueT &V) {
    TheMap.erase(V);
  }
  
  DenseSet &operator=(const DenseSet &RHS) {
    TheMap = RHS.TheMap;
    return *this;
  }
};

} // end namespace llvm

#endif
