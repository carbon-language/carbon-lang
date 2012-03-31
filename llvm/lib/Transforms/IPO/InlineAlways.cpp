//===- InlineAlways.cpp - Code to inline always_inline functions ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a custom inliner that handles only functions that
// are marked as "always inline".
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "inline"
#include "llvm/CallingConv.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/InlinerPass.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace llvm;

namespace {

  // AlwaysInliner only inlines functions that are mark as "always inline".
  class AlwaysInliner : public Inliner {
    InlineCostAnalyzer CA;
  public:
    // Use extremely low threshold.
    AlwaysInliner() : Inliner(ID, -2000000000, /*InsertLifetime*/true) {
      initializeAlwaysInlinerPass(*PassRegistry::getPassRegistry());
    }
    AlwaysInliner(bool InsertLifetime) : Inliner(ID, -2000000000,
                                                 InsertLifetime) {
      initializeAlwaysInlinerPass(*PassRegistry::getPassRegistry());
    }
    static char ID; // Pass identification, replacement for typeid
    InlineCost getInlineCost(CallSite CS) {
      Function *Callee = CS.getCalledFunction();
      // We assume indirect calls aren't calling an always-inline function.
      if (!Callee) return InlineCost::getNever();

      // We can't inline calls to external functions.
      // FIXME: We shouldn't even get here.
      if (Callee->isDeclaration()) return InlineCost::getNever();

      // Return never for anything not marked as always inline.
      if (!Callee->hasFnAttr(Attribute::AlwaysInline))
        return InlineCost::getNever();

      // We still have to check the inline cost in case there are reasons to
      // not inline which trump the always-inline attribute such as setjmp and
      // indirectbr.
      return CA.getInlineCost(CS, getInlineThreshold(CS));
    }
    void resetCachedCostInfo(Function *Caller) {
      CA.resetCachedCostInfo(Caller);
    }
    void growCachedCostInfo(Function* Caller, Function* Callee) {
      CA.growCachedCostInfo(Caller, Callee);
    }
    virtual bool doFinalization(CallGraph &CG) {
      return removeDeadFunctions(CG, /*AlwaysInlineOnly=*/true);
    }
    virtual bool doInitialization(CallGraph &CG);
    void releaseMemory() {
      CA.clear();
    }
  };
}

char AlwaysInliner::ID = 0;
INITIALIZE_PASS_BEGIN(AlwaysInliner, "always-inline",
                "Inliner for always_inline functions", false, false)
INITIALIZE_AG_DEPENDENCY(CallGraph)
INITIALIZE_PASS_END(AlwaysInliner, "always-inline",
                "Inliner for always_inline functions", false, false)

Pass *llvm::createAlwaysInlinerPass() { return new AlwaysInliner(); }

Pass *llvm::createAlwaysInlinerPass(bool InsertLifetime) {
  return new AlwaysInliner(InsertLifetime);
}

// doInitialization - Initializes the vector of functions that have not
// been annotated with the "always inline" attribute.
bool AlwaysInliner::doInitialization(CallGraph &CG) {
  CA.setTargetData(getAnalysisIfAvailable<TargetData>());
  return false;
}
