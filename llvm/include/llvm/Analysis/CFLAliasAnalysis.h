//===- CFLAliasAnalysis.h - CFL-Based Alias Analysis Interface ---*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This is the interface for LLVM's primary stateless and local alias analysis.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CFLALIASANALYSIS_H
#define LLVM_ANALYSIS_CFLALIASANALYSIS_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include <forward_list>

namespace llvm {

class CFLAliasAnalysis : public ImmutablePass, public AliasAnalysis {
  struct FunctionInfo;

  struct FunctionHandle final : public CallbackVH {
    FunctionHandle(Function *Fn, CFLAliasAnalysis *CFLAA)
        : CallbackVH(Fn), CFLAA(CFLAA) {
      assert(Fn != nullptr);
      assert(CFLAA != nullptr);
    }

    void deleted() override { removeSelfFromCache(); }
    void allUsesReplacedWith(Value *) override { removeSelfFromCache(); }

  private:
    CFLAliasAnalysis *CFLAA;

    void removeSelfFromCache() {
      assert(CFLAA != nullptr);
      auto *Val = getValPtr();
      CFLAA->evict(cast<Function>(Val));
      setValPtr(nullptr);
    }
  };

  /// \brief Cached mapping of Functions to their StratifiedSets.
  /// If a function's sets are currently being built, it is marked
  /// in the cache as an Optional without a value. This way, if we
  /// have any kind of recursion, it is discernable from a function
  /// that simply has empty sets.
  DenseMap<Function *, Optional<FunctionInfo>> Cache;
  std::forward_list<FunctionHandle> Handles;

public:
  static char ID;

  CFLAliasAnalysis();
  ~CFLAliasAnalysis() override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  void *getAdjustedAnalysisPointer(const void *ID) override;

  /// \brief Inserts the given Function into the cache.
  void scan(Function *Fn);

  void evict(Function *Fn);

  /// \brief Ensures that the given function is available in the cache.
  /// Returns the appropriate entry from the cache.
  const Optional<FunctionInfo> &ensureCached(Function *Fn);

  AliasResult query(const MemoryLocation &LocA, const MemoryLocation &LocB);

  AliasResult alias(const MemoryLocation &LocA,
                    const MemoryLocation &LocB) override {
    if (LocA.Ptr == LocB.Ptr) {
      if (LocA.Size == LocB.Size) {
        return MustAlias;
      } else {
        return PartialAlias;
      }
    }

    // Comparisons between global variables and other constants should be
    // handled by BasicAA.
    // TODO: ConstantExpr handling -- CFLAA may report NoAlias when comparing
    // a GlobalValue and ConstantExpr, but every query needs to have at least
    // one Value tied to a Function, and neither GlobalValues nor ConstantExprs
    // are.
    if (isa<Constant>(LocA.Ptr) && isa<Constant>(LocB.Ptr)) {
      return AliasAnalysis::alias(LocA, LocB);
    }

    AliasResult QueryResult = query(LocA, LocB);
    if (QueryResult == MayAlias)
      return AliasAnalysis::alias(LocA, LocB);

    return QueryResult;
  }

  bool doInitialization(Module &M) override;

private:
  FunctionInfo buildSetsFrom(Function *F);
};

//===--------------------------------------------------------------------===//
//
// createCFLAliasAnalysisPass - This pass implements a set-based approach to
// alias analysis.
//
ImmutablePass *createCFLAliasAnalysisPass();

}

#endif
