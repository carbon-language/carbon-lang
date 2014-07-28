//===- llvm/IR/UseListOrder.h - LLVM Use List Order functions ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file has functions to modify the use-list order and to verify that it
// doesn't change after serialization.
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

public:
  UseListShuffleVector() : Size(0) {}
  UseListShuffleVector(UseListShuffleVector &&X) {
    std::memcpy(this, &X, sizeof(UseListShuffleVector));
    X.Size = 0;
  }
  explicit UseListShuffleVector(size_t Size) : Size(Size) {
    if (!isSmall())
      Storage.Ptr = new unsigned[Size];
  }
  ~UseListShuffleVector() {
    if (!isSmall())
      delete Storage.Ptr;
  }

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
};

typedef std::vector<UseListOrder> UseListOrderStack;

/// \brief Whether to preserve use-list ordering.
bool shouldPreserveBitcodeUseListOrder();
bool shouldPreserveAssemblyUseListOrder();

/// \brief Shuffle all use-lists in a module.
///
/// Adds \c SeedOffset to the default seed for the random number generator.
void shuffleUseLists(Module &M, unsigned SeedOffset = 0);

} // end namespace llvm

#endif
