//==- GRBlockCounter.h - ADT for counting block visits -------------*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines GRBlockCounter, an abstract data type used to count
//  the number of times a given block has been visited along a path
//  analyzed by GRCoreEngine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_GRBLOCKCOUNTER
#define LLVM_CLANG_ANALYSIS_GRBLOCKCOUNTER

namespace llvm {
  class BumpPtrAllocator;
}

namespace clang {

class GRBlockCounter {
  void* Data;

  GRBlockCounter(void* D) : Data(D) {}

public:
  GRBlockCounter() : Data(0) {}

  unsigned getNumVisited(unsigned BlockID) const;

  class Factory {
    void* F;
  public:
    Factory(llvm::BumpPtrAllocator& Alloc);
    ~Factory();

    GRBlockCounter GetEmptyCounter();
    GRBlockCounter IncrementCount(GRBlockCounter BC, unsigned BlockID);
  };

  friend class Factory;
};

} // end clang namespace

#endif
