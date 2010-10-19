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
#include "llvm/ADT/SmallPtrSet.h"

using namespace llvm;

namespace {

  // AlwaysInliner only inlines functions that are mark as "always inline".
  class AlwaysInliner : public Inliner {
    // Functions that are never inlined
    SmallPtrSet<const Function*, 16> NeverInline; 
    InlineCostAnalyzer CA;
  public:
    // Use extremely low threshold. 
    AlwaysInliner() : Inliner(ID, -2000000000) {
      initializeAlwaysInlinerPass(*PassRegistry::getPassRegistry());
    }
    static char ID; // Pass identification, replacement for typeid
    InlineCost getInlineCost(CallSite CS) {
      return CA.getInlineCost(CS, NeverInline);
    }
    float getInlineFudgeFactor(CallSite CS) {
      return CA.getInlineFudgeFactor(CS);
    }
    void resetCachedCostInfo(Function *Caller) {
      CA.resetCachedCostInfo(Caller);
    }
    void growCachedCostInfo(Function* Caller, Function* Callee) {
      CA.growCachedCostInfo(Caller, Callee);
    }
    virtual bool doFinalization(CallGraph &CG) { 
      return removeDeadFunctions(CG, &NeverInline); 
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

// doInitialization - Initializes the vector of functions that have not 
// been annotated with the "always inline" attribute.
bool AlwaysInliner::doInitialization(CallGraph &CG) {
  Module &M = CG.getModule();
  
  for (Module::iterator I = M.begin(), E = M.end();
       I != E; ++I)
    if (!I->isDeclaration() && !I->hasFnAttr(Attribute::AlwaysInline))
      NeverInline.insert(I);

  return false;
}
