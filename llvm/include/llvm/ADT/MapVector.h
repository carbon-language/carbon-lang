//===- llvm/ADT/MapVector.h - Map with deterministic value order *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a map that also provides access to all stored values
// in a deterministic order via the getValues method. Note that the iteration
// order itself is just the DenseMap order and not deterministic. The interface
// is purposefully minimal.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_MAPVECTOR_H
#define LLVM_ADT_MAPVECTOR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include <vector>

namespace llvm {

/// This class implements a map that also provides access to all stored values
/// in a deterministic order. The values are kept in a std::vector and the
/// mapping is done with DenseMap from Keys to indexes in that vector.
template<typename KeyT, typename ValueT>
class MapVector {
  typedef llvm::DenseMap<KeyT, unsigned> MapType;
  typedef std::vector<ValueT> VectorType;
  typedef typename VectorType::size_type SizeType;

  MapType Map;
  VectorType Vector;

public:
  // The keys and values are not stored close to each other, so the iterator
  // operator->() cannot return a pointer to a std::pair like a DenseMap does.
  // Instead it returns a FakePair that contains references to Key and Value.
  // This lets code using this to look the same as if using a regular DenseMap.
  template<bool IsConst>
  struct FakePair {
    typedef typename conditional<IsConst, const ValueT, ValueT>::type VT;
    const KeyT &first;
    VT &second;
    FakePair(const KeyT &K, VT &V) : first(K), second(V) {
    }
    FakePair *operator->() {
      return this;
    }
  };

  template<bool IsConst>
  class IteratorTemplate {
    typedef typename MapType::const_iterator WrappedIteratorType;
    WrappedIteratorType WrappedI;
    typedef
      typename conditional<IsConst, const VectorType, VectorType>::type VT;
    VT &VecRef;
    typedef FakePair<IsConst> PairType;
    friend class IteratorTemplate<true>;

  public:
    IteratorTemplate(WrappedIteratorType I, VT &V) :
      WrappedI(I), VecRef(V) {
    }

    // If IsConst is true this is a converting constructor from iterator to
    // const_iterator and the default copy constructor is used.
    // Otherwise this is a copy constructor for iterator.
    IteratorTemplate(const IteratorTemplate<false>& I) :
      WrappedI(I.WrappedI), VecRef(I.VecRef) {
    }

    bool operator!=(const IteratorTemplate &RHS) const {
      return WrappedI != RHS.WrappedI;
    }

    IteratorTemplate &operator++() {  // Preincrement
      ++WrappedI;
      return *this;
    }

    PairType operator->() {
      unsigned Pos = WrappedI->second;
      PairType Ret(WrappedI->first, VecRef[Pos]);
      return Ret;
    }
  };

  typedef IteratorTemplate<false> iterator;
  typedef IteratorTemplate<true> const_iterator;

  SizeType size() const {
    return Vector.size();
  }

  iterator begin() {
    return iterator(Map.begin(), this->Vector);
  }

  const_iterator begin() const {
    return const_iterator(Map.begin(), this->Vector);
  }

  iterator end() {
    return iterator(Map.end(), this->Vector);
  }

  const_iterator end() const {
    return const_iterator(Map.end(), this->Vector);
  }

  bool empty() const {
    return Map.empty();
  }

  typedef typename VectorType::iterator value_iterator;
  typedef typename VectorType::const_iterator const_value_iterator;

  value_iterator value_begin() {
    return Vector.begin();
  }

  const_value_iterator value_begin() const {
    return Vector.begin();
  }

  value_iterator value_end() {
    return Vector.end();
  }

  const_value_iterator value_end() const {
    return Vector.end();
  }

  ValueT &operator[](const KeyT &Key) {
    std::pair<KeyT, unsigned> Pair = std::make_pair(Key, 0);
    std::pair<typename MapType::iterator, bool> Result = Map.insert(Pair);
    unsigned &I = Result.first->second;
    if (Result.second) {
      Vector.push_back(ValueT());
      I = Vector.size() - 1;
    }
    return Vector[I];
  }
};

}

#endif
