//===- TypeBasedAliasAnalysis.h - Type-Based Alias Analysis -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This is the interface for a metadata-based TBAA. See the source file for
/// details on the algorithm.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_TYPEBASEDALIASANALYSIS_H
#define LLVM_ANALYSIS_TYPEBASEDALIASANALYSIS_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Pass.h"

namespace llvm {

/// TypeBasedAliasAnalysis - This is a simple alias analysis
/// implementation that uses TypeBased to answer queries.
class TypeBasedAliasAnalysis : public ImmutablePass, public AliasAnalysis {
public:
  static char ID; // Class identification, replacement for typeinfo
  TypeBasedAliasAnalysis() : ImmutablePass(ID) {
    initializeTypeBasedAliasAnalysisPass(*PassRegistry::getPassRegistry());
  }

  bool doInitialization(Module &M) override;

  /// getAdjustedAnalysisPointer - This method is used when a pass implements
  /// an analysis interface through multiple inheritance.  If needed, it
  /// should override this to adjust the this pointer as needed for the
  /// specified pass info.
  void *getAdjustedAnalysisPointer(const void *PI) override {
    if (PI == &AliasAnalysis::ID)
      return (AliasAnalysis *)this;
    return this;
  }

  bool Aliases(const MDNode *A, const MDNode *B) const;
  bool PathAliases(const MDNode *A, const MDNode *B) const;

private:
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  AliasResult alias(const MemoryLocation &LocA,
                    const MemoryLocation &LocB) override;
  bool pointsToConstantMemory(const MemoryLocation &Loc, bool OrLocal) override;
  FunctionModRefBehavior getModRefBehavior(ImmutableCallSite CS) override;
  FunctionModRefBehavior getModRefBehavior(const Function *F) override;
  ModRefInfo getModRefInfo(ImmutableCallSite CS,
                           const MemoryLocation &Loc) override;
  ModRefInfo getModRefInfo(ImmutableCallSite CS1,
                           ImmutableCallSite CS2) override;
};

//===--------------------------------------------------------------------===//
//
// createTypeBasedAliasAnalysisPass - This pass implements metadata-based
// type-based alias analysis.
//
ImmutablePass *createTypeBasedAliasAnalysisPass();

}

#endif
