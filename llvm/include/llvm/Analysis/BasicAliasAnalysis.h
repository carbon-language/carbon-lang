//===- llvm/Analysis/BasicAliasAnalysis.h - Alias Analysis Impl -*- C++ -*-===//
//
// This file defines the generic AliasAnalysis interface, which is used as the
// common interface used by all clients of alias analysis information, and
// implemented by all alias analysis implementations.
//
// Implementations of this interface must implement the various virtual methods,
// which automatically provides functionality for the entire suite of client
// APIs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_BASIC_ALIAS_ANALYSIS_H
#define LLVM_ANALYSIS_BASIC_ALIAS_ANALYSIS_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Pass.h"

struct BasicAliasAnalysis : public FunctionPass, public AliasAnalysis {
  
  // Pass Implementation stuff.  This isn't much of a pass.
  //
  bool runOnFunction(Function &) { return false; }
    
  // getAnalysisUsage - Does not modify anything.
  //
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
  
  // alias - This is the only method here that does anything interesting...
  //
  Result alias(const Value *V1, const Value *V2) const;
    
  // canCallModify - We are not interprocedural, so we do nothing exciting.
  //
  Result canCallModify(const CallInst &CI, const Value *Ptr) const {
    return MayAlias;
  }
    
  // canInvokeModify - We are not interprocedural, so we do nothing exciting.
  //
  Result canInvokeModify(const InvokeInst &I, const Value *Ptr) const {
    return MayAlias;  // We are not interprocedural
  }
};

#endif
