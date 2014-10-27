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
  typedef DenseMap<ValueT, char, ValueInfoT> MapTy;
  MapTy TheMap;
public:
  typedef ValueT key_type;
  typedef ValueT value_type;
  typedef unsigned size_type;

  explicit DenseSet(unsigned NumInitBuckets = 0) : TheMap(NumInitBuckets) {}

  bool empty() const { return TheMap.empty(); }
  size_type size() const { return TheMap.size(); }
  size_t getMemorySize() const { return TheMap.getMemorySize(); }

  /// Grow the DenseSet so that it has at least Size buckets. Will not shrink
  /// the Size of the set.
  void resize(size_t Size) { TheMap.resize(Size); }

  void clear() {
    TheMap.clear();
  }

  /// Return 1 if the specified key is in the set, 0 otherwise.
  size_type count(const ValueT &V) const {
    return TheMap.count(V);
  }

  bool erase(const ValueT &V) {
    return TheMap.erase(V);
  }

  void swap(DenseSet& RHS) {
    TheMap.swap(RHS.TheMap);
  }

  // Iterators.

  class Iterator {
    typename MapTy::iterator I;
    friend class DenseSet;
  public:
    typedef typename MapTy::iterator::difference_type difference_type;
    typedef ValueT value_type;
    typedef value_type *pointer;
    typedef value_type &reference;
    typedef std::forward_iterator_tag iterator_category;

    Iterator(const typename MapTy::iterator &i) : I(i) {}

    ValueT& operator*() { return I->first; }
    ValueT* operator->() { return &I->first; }

    Iterator& operator++() { ++I; return *this; }
    bool operator==(const Iterator& X) const { return I == X.I; }
    bool operator!=(const Iterator& X) const { return I != X.I; }
  };

  class ConstIterator {
    typename MapTy::const_iterator I;
    friend class DenseSet;
  public:
    typedef typename MapTy::const_iterator::difference_type difference_type;
    typedef ValueT value_type;
    typedef value_type *pointer;
    typedef value_type &reference;
    typedef std::forward_iterator_tag iterator_category;

    ConstIterator(const typename MapTy::const_iterator &i) : I(i) {}

    const ValueT& operator*() { return I->first; }
    const ValueT* operator->() { return &I->first; }

    ConstIterator& operator++() { ++I; return *this; }
    bool operator==(const ConstIterator& X) const { return I == X.I; }
    bool operator!=(const ConstIterator& X) const { return I != X.I; }
  };

  typedef Iterator      iterator;
  typedef ConstIterator const_iterator;

  iterator begin() { return Iterator(TheMap.begin()); }
  iterator end() { return Iterator(TheMap.end()); }

  const_iterator begin() const { return ConstIterator(TheMap.begin()); }
  const_iterator end() const { return ConstIterator(TheMap.end()); }

  iterator find(const ValueT &V) { return Iterator(TheMap.find(V)); }

  /// Alternative version of find() which allows a different, and possibly less
  /// expensive, key type.
  /// The DenseMapInfo is responsible for supplying methods
  /// getHashValue(LookupKeyT) and isEqual(LookupKeyT, KeyT) for each key type
  /// used.
  template <class LookupKeyT>
  iterator find_as(const LookupKeyT &Val) {
    return Iterator(TheMap.find_as(Val));
  }
  template <class LookupKeyT>
  const_iterator find_as(const LookupKeyT &Val) const {
    return ConstIterator(TheMap.find_as(Val));
  }

  void erase(Iterator I) { return TheMap.erase(I.I); }
  void erase(ConstIterator CI) { return TheMap.erase(CI.I); }

  std::pair<iterator, bool> insert(const ValueT &V) {
    return TheMap.insert(std::make_pair(V, 0));
  }
  
  // Range insertion of values.
  template<typename InputIt>
  void insert(InputIt I, InputIt E) {
    for (; I != E; ++I)
      insert(*I);
  }
};

} // end namespace llvm

#endif
