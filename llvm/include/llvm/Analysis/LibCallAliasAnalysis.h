//===- LibCallAliasAnalysis.h - Implement AliasAnalysis for libcalls ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the LibCallAliasAnalysis class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LIBCALLALIASANALYSIS_H
#define LLVM_ANALYSIS_LIBCALLALIASANALYSIS_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

namespace llvm {

class LibCallInfo;
struct LibCallFunctionInfo;

/// Alias analysis driven from LibCallInfo.
struct LibCallAliasAnalysis : public FunctionPass, public AliasAnalysis {
  static char ID; // Class identification

  LibCallInfo *LCI;

  explicit LibCallAliasAnalysis(LibCallInfo *LC = nullptr)
      : FunctionPass(ID), LCI(LC) {
    initializeLibCallAliasAnalysisPass(*PassRegistry::getPassRegistry());
  }
  explicit LibCallAliasAnalysis(char &ID, LibCallInfo *LC)
      : FunctionPass(ID), LCI(LC) {
    initializeLibCallAliasAnalysisPass(*PassRegistry::getPassRegistry());
  }
  ~LibCallAliasAnalysis() override;

  ModRefInfo getModRefInfo(ImmutableCallSite CS,
                           const MemoryLocation &Loc) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnFunction(Function &F) override;

  /// This method is used when a pass implements an analysis interface through
  /// multiple inheritance.
  ///
  /// If needed, it should override this to adjust the this pointer as needed
  /// for the specified pass info.
  void *getAdjustedAnalysisPointer(const void *PI) override {
    if (PI == &AliasAnalysis::ID)
      return (AliasAnalysis *)this;
    return this;
  }

private:
  ModRefInfo AnalyzeLibCallDetails(const LibCallFunctionInfo *FI,
                                   ImmutableCallSite CS,
                                   const MemoryLocation &Loc);
};

/// Create an alias analysis pass that knows about the semantics of a set of
/// libcalls specified by LCI.
///
/// The newly constructed pass takes ownership of the pointer that is provided.
FunctionPass *createLibCallAliasAnalysisPass(LibCallInfo *LCI);

} // End of llvm namespace

#endif
