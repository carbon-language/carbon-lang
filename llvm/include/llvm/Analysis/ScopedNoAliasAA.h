//===- ScopedNoAliasAA.h - Scoped No-Alias Alias Analysis -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This is the interface for a metadata-based scoped no-alias analysis.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SCOPEDNOALIASAA_H
#define LLVM_ANALYSIS_SCOPEDNOALIASAA_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

namespace llvm {

/// ScopedNoAliasAA - This is a simple alias analysis
/// implementation that uses scoped-noalias metadata to answer queries.
class ScopedNoAliasAA : public ImmutablePass, public AliasAnalysis {
public:
  static char ID; // Class identification, replacement for typeinfo
  ScopedNoAliasAA() : ImmutablePass(ID) {
    initializeScopedNoAliasAAPass(*PassRegistry::getPassRegistry());
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

protected:
  bool mayAliasInScopes(const MDNode *Scopes, const MDNode *NoAlias) const;
  void collectMDInDomain(const MDNode *List, const MDNode *Domain,
                         SmallPtrSetImpl<const MDNode *> &Nodes) const;

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
// createScopedNoAliasAAPass - This pass implements metadata-based
// scoped noalias analysis.
//
ImmutablePass *createScopedNoAliasAAPass();

}

#endif
