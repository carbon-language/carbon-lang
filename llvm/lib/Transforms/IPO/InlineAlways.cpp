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
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/InlinerPass.h"
#include "llvm/Transforms/Utils/InlineCost.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace llvm;

namespace {

  // AlwaysInliner only inlines functions that are mark as "always inline".
  class VISIBILITY_HIDDEN AlwaysInliner : public Inliner {
    // Functions that are never inlined
    SmallPtrSet<const Function*, 16> NeverInline; 
    InlineCostAnalyzer CA;
  public:
    // Use extremely low threshold. 
    AlwaysInliner() : Inliner(&ID, -2000000000) {}
    static char ID; // Pass identification, replacement for typeid
    int getInlineCost(CallSite CS) {
      return CA.getInlineCost(CS, NeverInline);
    }
    float getInlineFudgeFactor(CallSite CS) {
      return CA.getInlineFudgeFactor(CS);
    }
    virtual bool doInitialization(CallGraph &CG);
  };
}

char AlwaysInliner::ID = 0;
static RegisterPass<AlwaysInliner>
X("always-inline", "Inliner that handles always_inline functions");

Pass *llvm::createAlwaysInlinerPass() { return new AlwaysInliner(); }

// doInitialization - Initializes the vector of functions that have not 
// been annotated with the "always inline" attribute.
bool AlwaysInliner::doInitialization(CallGraph &CG) {
  
  Module &M = CG.getModule();
  
  for (Module::iterator I = M.begin(), E = M.end();
       I != E; ++I)
    if (!I->isDeclaration() && !I->hasNote(ParamAttr::FN_NOTE_AlwaysInline))
      NeverInline.insert(I);

  return false;
}

