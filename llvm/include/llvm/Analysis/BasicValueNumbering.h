//===- llvm/Analysis/BasicValueNumbering.h - Value #'ing Impl ---*- C++ -*-===//
//
// This file defines the default implementation of the Value Numbering
// interface, which uses the SSA value graph to find lexically identical
// expressions.  This does not require any computation ahead of time, so it is a
// very fast default implementation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_BASIC_VALUE_NUMBERING_H
#define LLVM_ANALYSIS_BASIC_VALUE_NUMBERING_H

#include "llvm/Analysis/ValueNumbering.h"
#include "llvm/Pass.h"

struct BasicValueNumbering : public FunctionPass, public ValueNumbering {
  
  /// Pass Implementation stuff.  This isn't much of a pass.
  ///
  bool runOnFunction(Function &) { return false; }
    
  /// getAnalysisUsage - Does not modify anything.
  ///
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
  
  /// getEqualNumberNodes - Return nodes with the same value number as the
  /// specified Value.  This fills in the argument vector with any equal values.
  ///
  /// This is where our implementation is.
  ///
  virtual void getEqualNumberNodes(Value *V1,
                                   std::vector<Value*> &RetVals) const;
};

#endif
