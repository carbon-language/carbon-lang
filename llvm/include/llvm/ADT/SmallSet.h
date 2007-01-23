//===- llvm/ADT/SmallSet.h - 'Normally small' sets --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the SmallSet class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SMALLSET_H
#define LLVM_ADT_SMALLSET_H

#include "llvm/ADT/SmallVector.h"

namespace llvm {

/// SmallSet - This maintains a set of unique values, optimizing for the case
/// when the set is small (less than N).  In this case, the set can be
/// maintained with no mallocs.
///
/// Note that this set does not guarantee that the elements in the set will be
/// ordered.
template <typename T, unsigned N>
class SmallSet {
  SmallVector<T, N> Vector;
  typedef typename SmallVector<T, N>::iterator mutable_iterator;
public:
  SmallSet() {}

  // Support iteration.
  typedef typename SmallVector<T, N>::const_iterator iterator;
  typedef typename SmallVector<T, N>::const_iterator const_iterator;
  
  iterator begin() const { return Vector.begin(); }
  iterator end() const { return Vector.end(); }
  
  bool empty() const { return Vector.empty(); }
  unsigned size() const { return Vector.size(); }
  
  iterator find(const T &V) const {
    for (iterator I = begin(), E = end(); I != E; ++I)
      if (*I == V)
        return I;
    return end();
  }
  
  /// count - Return true if the element is in the set.
  unsigned count(const T &V) const {
    // Since the collection is small, just do a linear search.
    return find(V) != end();
  }
  
  /// insert - Insert an element into the set if it isn't already there.
  std::pair<iterator,bool> insert(const T &V) {
    iterator I = find(V);
    if (I == end())    // Don't reinsert if it already exists.
      return std::make_pair(I, false);
    Vector.push_back(V);
    return std::make_pair(end()-1, true);
  }
  
  void erase(const T &V) {
    for (mutable_iterator I = Vector.begin(), E = Vector.end(); I != E; ++I)
      if (*I == V) {
        Vector.erase(I);
        return;
      }
  }
  
  void clear() {
    Vector.clear();
  }
  
};


} // end namespace llvm

#endif
