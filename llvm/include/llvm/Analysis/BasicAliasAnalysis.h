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

class GetElementPtrInst;

struct BasicAliasAnalysis : public ImmutablePass, public AliasAnalysis {

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AliasAnalysis::getAnalysisUsage(AU);
  }

  virtual void initializePass();

  // alias - This is the only method here that does anything interesting...
  //
  AliasResult alias(const Value *V1, unsigned V1Size,
                    const Value *V2, unsigned V2Size);
private:
  // CheckGEPInstructions - Check two GEP instructions of compatible types and
  // equal number of arguments.  This checks to see if the index expressions
  // preclude the pointers from aliasing...
  AliasResult CheckGEPInstructions(GetElementPtrInst *GEP1, unsigned G1Size,
                                   GetElementPtrInst *GEP2, unsigned G2Size);
};

#endif
