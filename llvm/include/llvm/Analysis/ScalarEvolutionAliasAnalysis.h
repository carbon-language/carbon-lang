//===- ScalarEvolutionAliasAnalysis.h - SCEV-based AA -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This is the interface for a SCEV-based alias analysis.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SCALAREVOLUTIONALIASANALYSIS_H
#define LLVM_ANALYSIS_SCALAREVOLUTIONALIASANALYSIS_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

namespace llvm {

/// ScalarEvolutionAliasAnalysis - This is a simple alias analysis
/// implementation that uses ScalarEvolution to answer queries.
class ScalarEvolutionAliasAnalysis : public FunctionPass, public AliasAnalysis {
  ScalarEvolution *SE;

public:
  static char ID; // Class identification, replacement for typeinfo
  ScalarEvolutionAliasAnalysis() : FunctionPass(ID), SE(nullptr) {
    initializeScalarEvolutionAliasAnalysisPass(
        *PassRegistry::getPassRegistry());
  }

  /// getAdjustedAnalysisPointer - This method is used when a pass implements
  /// an analysis interface through multiple inheritance.  If needed, it
  /// should override this to adjust the this pointer as needed for the
  /// specified pass info.
  void *getAdjustedAnalysisPointer(AnalysisID PI) override {
    if (PI == &AliasAnalysis::ID)
      return (AliasAnalysis *)this;
    return this;
  }

private:
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnFunction(Function &F) override;
  AliasResult alias(const MemoryLocation &LocA,
                    const MemoryLocation &LocB) override;

  Value *GetBaseValue(const SCEV *S);
};

//===--------------------------------------------------------------------===//
//
// createScalarEvolutionAliasAnalysisPass - This pass implements a simple
// alias analysis using ScalarEvolution queries.
//
FunctionPass *createScalarEvolutionAliasAnalysisPass();

}

#endif
