//===- llvm/Analysis/BasicAliasAnalysis.h - Alias Analysis Impl -*- C++ -*-===//
//
// This file defines the default implementation of the Alias Analysis interface
// that simply implements a few identities (two different globals cannot alias,
// etc), but otherwise does no analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_BASIC_ALIAS_ANALYSIS_H
#define LLVM_ANALYSIS_BASIC_ALIAS_ANALYSIS_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Pass.h"

struct BasicAliasAnalysis : public ImmutablePass, public AliasAnalysis {

  // alias - This is the only method here that does anything interesting...
  //
  Result alias(const Value *V1, const Value *V2) const;
    
  /// canCallModify - We are not interprocedural, so we do nothing exciting.
  ///
  Result canCallModify(const CallInst &CI, const Value *Ptr) const {
    return MayAlias;
  }
    
  /// canInvokeModify - We are not interprocedural, so we do nothing exciting.
  ///
  Result canInvokeModify(const InvokeInst &I, const Value *Ptr) const {
    return MayAlias;  // We are not interprocedural
  }
};

#endif
