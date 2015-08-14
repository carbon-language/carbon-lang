//===- GlobalsModRef.h - Simple Mod/Ref AA for Globals ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This is the interface for a simple mod/ref and alias analysis over globlas.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_GLOBALSMODREF_H
#define LLVM_ANALYSIS_GLOBALSMODREF_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include <list>

namespace llvm {

/// GlobalsModRef - The actual analysis pass.
class GlobalsModRef : public ModulePass, public AliasAnalysis {
  class FunctionInfo;

  /// The globals that do not have their addresses taken.
  SmallPtrSet<const GlobalValue *, 8> NonAddressTakenGlobals;

  /// IndirectGlobals - The memory pointed to by this global is known to be
  /// 'owned' by the global.
  SmallPtrSet<const GlobalValue *, 8> IndirectGlobals;

  /// AllocsForIndirectGlobals - If an instruction allocates memory for an
  /// indirect global, this map indicates which one.
  DenseMap<const Value *, const GlobalValue *> AllocsForIndirectGlobals;

  /// For each function, keep track of what globals are modified or read.
  DenseMap<const Function *, FunctionInfo> FunctionInfos;

  /// Handle to clear this analysis on deletion of values.
  struct DeletionCallbackHandle final : CallbackVH {
    GlobalsModRef &GMR;
    std::list<DeletionCallbackHandle>::iterator I;

    DeletionCallbackHandle(GlobalsModRef &GMR, Value *V)
        : CallbackVH(V), GMR(GMR) {}

    void deleted() override;
  };

  /// List of callbacks for globals being tracked by this analysis. Note that
  /// these objects are quite large, but we only anticipate having one per
  /// global tracked by this analysis. There are numerous optimizations we
  /// could perform to the memory utilization here if this becomes a problem.
  std::list<DeletionCallbackHandle> Handles;

public:
  static char ID;
  GlobalsModRef();

  bool runOnModule(Module &M) override {
    InitializeAliasAnalysis(this, &M.getDataLayout());

    // Find non-addr taken globals.
    AnalyzeGlobals(M);

    // Propagate on CG.
    AnalyzeCallGraph(getAnalysis<CallGraphWrapperPass>().getCallGraph(), M);
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AliasAnalysis::getAnalysisUsage(AU);
    AU.addRequired<CallGraphWrapperPass>();
    AU.setPreservesAll(); // Does not transform code
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

  //------------------------------------------------
  // Implement the AliasAnalysis API
  //
  AliasResult alias(const MemoryLocation &LocA,
                    const MemoryLocation &LocB) override;
  ModRefInfo getModRefInfo(ImmutableCallSite CS,
                           const MemoryLocation &Loc) override;
  ModRefInfo getModRefInfo(ImmutableCallSite CS1,
                           ImmutableCallSite CS2) override {
    return AliasAnalysis::getModRefInfo(CS1, CS2);
  }

  /// getModRefBehavior - Return the behavior of the specified function if
  /// called from the specified call site.  The call site may be null in which
  /// case the most generic behavior of this function should be returned.
  FunctionModRefBehavior getModRefBehavior(const Function *F) override;

  /// getModRefBehavior - Return the behavior of the specified function if
  /// called from the specified call site.  The call site may be null in which
  /// case the most generic behavior of this function should be returned.
  FunctionModRefBehavior getModRefBehavior(ImmutableCallSite CS) override;

private:
  FunctionInfo *getFunctionInfo(const Function *F);

  void AnalyzeGlobals(Module &M);
  void AnalyzeCallGraph(CallGraph &CG, Module &M);
  bool AnalyzeUsesOfPointer(Value *V,
                            SmallPtrSetImpl<Function *> *Readers = nullptr,
                            SmallPtrSetImpl<Function *> *Writers = nullptr,
                            GlobalValue *OkayStoreDest = nullptr);
  bool AnalyzeIndirectGlobalMemory(GlobalValue *GV);

  bool isNonEscapingGlobalNoAlias(const GlobalValue *GV, const Value *V);
};

//===--------------------------------------------------------------------===//
//
// createGlobalsModRefPass - This pass provides alias and mod/ref info for
// global values that do not have their addresses taken.
//
Pass *createGlobalsModRefPass();

}

#endif
