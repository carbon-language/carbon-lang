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

#ifndef LLVM_CLANG_GR_GRBLOCKCOUNTER
#define LLVM_CLANG_GR_GRBLOCKCOUNTER

namespace llvm {
  class BumpPtrAllocator;
}

namespace clang {

class StackFrameContext;

namespace GR {

class GRBlockCounter {
  void* Data;

  GRBlockCounter(void* D) : Data(D) {}

public:
  GRBlockCounter() : Data(0) {}

  unsigned getNumVisited(const StackFrameContext *CallSite, 
                         unsigned BlockID) const;

  class Factory {
    void* F;
  public:
    Factory(llvm::BumpPtrAllocator& Alloc);
    ~Factory();

    GRBlockCounter GetEmptyCounter();
    GRBlockCounter IncrementCount(GRBlockCounter BC, 
                                  const StackFrameContext *CallSite,
                                  unsigned BlockID);
  };

  friend class Factory;
};

} // end GR namespace

} // end clang namespace

#endif
