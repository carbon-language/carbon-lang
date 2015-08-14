//===- AliasAnalysisCounter.h - Alias Analysis Query Counter ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This declares an alias analysis which counts and prints queries made
/// through it. By inserting this between other AAs you can track when specific
/// layers of LLVM's AA get queried.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ALIASANALYSISCOUNTER_H
#define LLVM_ANALYSIS_ALIASANALYSISCOUNTER_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

namespace llvm {

class AliasAnalysisCounter : public ModulePass, public AliasAnalysis {
  unsigned No, May, Partial, Must;
  unsigned NoMR, JustRef, JustMod, MR;
  Module *M;

public:
  static char ID; // Class identification, replacement for typeinfo

  AliasAnalysisCounter();
  ~AliasAnalysisCounter() override;

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// getAdjustedAnalysisPointer - This method is used when a pass implements
  /// an analysis interface through multiple inheritance.  If needed, it
  /// should override this to adjust the this pointer as needed for the
  /// specified pass info.
  void *getAdjustedAnalysisPointer(AnalysisID PI) override;

  // Forwarding functions: just delegate to a real AA implementation, counting
  // the number of responses...
  AliasResult alias(const MemoryLocation &LocA,
                    const MemoryLocation &LocB) override;

  ModRefInfo getModRefInfo(ImmutableCallSite CS,
                           const MemoryLocation &Loc) override;
};

//===--------------------------------------------------------------------===//
//
// createAliasAnalysisCounterPass - This pass counts alias queries and how the
// alias analysis implementation responds.
//
ModulePass *createAliasAnalysisCounterPass();

}

#endif
