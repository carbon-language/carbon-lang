//===- LazyValueInfo.h - Value constraint analysis --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for lazy computation of value constraint
// information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LIVEVALUES_H
#define LLVM_ANALYSIS_LIVEVALUES_H

#include "llvm/Pass.h"

namespace llvm {
  class Constant;
  class TargetData;
  class Value;
  
/// LazyValueInfo - This pass computes, caches, and vends lazy value constraint
/// information.
class LazyValueInfo : public FunctionPass {
  class TargetData *TD;
  void *PImpl;
public:
  static char ID;
  LazyValueInfo() : FunctionPass(&ID), PImpl(0) {}

  /// Tristate - This is used to return yes/no/dunno results.
  enum Tristate {
    Unknown = -1, No = 0, Yes = 1
  };
  
  
  // Public query interface.
  
  
  /// isEqual - Determine whether the specified value is known to be equal or
  /// not-equal to the specified constant at the end of the specified block.
  Tristate isEqual(Value *V, Constant *C, BasicBlock *BB);

  /// getConstant - Determine whether the specified value is known to be a
  /// constant at the end of the specified block.  Return null if not.
  Constant *getConstant(Value *V, BasicBlock *BB);
  
  
  // Implementation boilerplate.
  
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
  virtual void releaseMemory();
  virtual bool runOnFunction(Function &F);
};

}  // end namespace llvm

#endif

