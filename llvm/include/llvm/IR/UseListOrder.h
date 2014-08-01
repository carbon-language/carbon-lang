//===- llvm/IR/UseListOrder.h - LLVM Use List Order -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file has structures and command-line options for preserving use-list
// order.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_USELISTORDER_H
#define LLVM_IR_USELISTORDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>

namespace llvm {

class Module;
class Function;
class Value;

/// \brief Structure to hold a use-list shuffle vector.
///
/// Stores most use-lists locally, but large use-lists use an extra heap entry.
/// Costs two fewer pointers than the equivalent \a SmallVector.
class UseListShuffleVector {
  unsigned Size;
  union {
    unsigned *Ptr;
    unsigned Array[6];
  } Storage;

  bool isSmall() const { return Size <= 6; }
  unsigned *data() { return isSmall() ? Storage.Array : Storage.Ptr; }
  const unsigned *data() const {
    return isSmall() ? Storage.Array : Storage.Ptr;
  }

  void destroy() {
    if (!isSmall())
      delete[] Storage.Ptr;
  }
  void moveUnchecked(UseListShuffleVector &X) {
    std::memcpy(this, &X, sizeof(UseListShuffleVector));
    X.Size = 0;
  }

  UseListShuffleVector(const UseListShuffleVector &X) LLVM_DELETED_FUNCTION;
  UseListShuffleVector &
  operator=(const UseListShuffleVector &X) LLVM_DELETED_FUNCTION;

public:
  UseListShuffleVector() : Size(0) {}
  UseListShuffleVector(UseListShuffleVector &&X) { moveUnchecked(X); }
  UseListShuffleVector &operator=(UseListShuffleVector &&X) {
    destroy();
    moveUnchecked(X);
    return *this;
  }
  explicit UseListShuffleVector(size_t Size) : Size(Size) {
    if (!isSmall())
      Storage.Ptr = new unsigned[Size];
  }
  ~UseListShuffleVector() { destroy(); }

  typedef unsigned *iterator;
  typedef const unsigned *const_iterator;

  size_t size() const { return Size; }
  iterator begin() { return data(); }
  iterator end() { return begin() + size(); }
  const_iterator begin() const { return data(); }
  const_iterator end() const { return begin() + size(); }
  unsigned &operator[](size_t I) { return data()[I]; }
  unsigned operator[](size_t I) const { return data()[I]; }
};

/// \brief Structure to hold a use-list order.
struct UseListOrder {
  const Value *V;
  const Function *F;
  UseListShuffleVector Shuffle;

  UseListOrder(const Value *V, const Function *F, size_t ShuffleSize)
      : V(V), F(F), Shuffle(ShuffleSize) {}

  UseListOrder() : V(0), F(0) {}
  UseListOrder(UseListOrder &&X)
      : V(X.V), F(X.F), Shuffle(std::move(X.Shuffle)) {}
  UseListOrder &operator=(UseListOrder &&X) {
    V = X.V;
    F = X.F;
    Shuffle = std::move(X.Shuffle);
    return *this;
  }

private:
  UseListOrder(const UseListOrder &X) LLVM_DELETED_FUNCTION;
  UseListOrder &operator=(const UseListOrder &X) LLVM_DELETED_FUNCTION;
};

typedef std::vector<UseListOrder> UseListOrderStack;

/// \brief Whether to preserve use-list ordering.
bool shouldPreserveBitcodeUseListOrder();
bool shouldPreserveAssemblyUseListOrder();

} // end namespace llvm

#endif
