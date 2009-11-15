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
  LazyValueInfo(const LazyValueInfo&); // DO NOT IMPLEMENT.
  void operator=(const LazyValueInfo&); // DO NOT IMPLEMENT.
public:
  static char ID;
  LazyValueInfo() : FunctionPass(&ID), PImpl(0) {}
  ~LazyValueInfo() { assert(PImpl == 0 && "releaseMemory not called"); }

  /// Tristate - This is used to return true/false/dunno results.
  enum Tristate {
    Unknown = -1, False = 0, True = 1
  };
  
  
  // Public query interface.
  
  /// getPredicateOnEdge - Determine whether the specified value comparison
  /// with a constant is known to be true or false on the specified CFG edge.
  /// Pred is a CmpInst predicate.
  Tristate getPredicateOnEdge(unsigned Pred, Value *V, Constant *C,
                              BasicBlock *FromBB, BasicBlock *ToBB);
  
  
  /// getConstant - Determine whether the specified value is known to be a
  /// constant at the end of the specified block.  Return null if not.
  Constant *getConstant(Value *V, BasicBlock *BB);

  /// getConstantOnEdge - Determine whether the specified value is known to be a
  /// constant on the specified edge.  Return null if not.
  Constant *getConstantOnEdge(Value *V, BasicBlock *FromBB, BasicBlock *ToBB);
  
  
  // Implementation boilerplate.
  
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
  virtual void releaseMemory();
  virtual bool runOnFunction(Function &F);
};

}  // end namespace llvm

#endif

