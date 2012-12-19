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
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/CallingConv.h"
#include "llvm/DataLayout.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Transforms/IPO/InlinerPass.h"
#include "llvm/Type.h"

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
    virtual InlineCost getInlineCost(CallSite CS);

    using llvm::Pass::doInitialization;
    using llvm::Pass::doFinalization;

    virtual bool doFinalization(CallGraph &CG) {
      return removeDeadFunctions(CG, /*AlwaysInlineOnly=*/true);
    }
    virtual bool doInitialization(CallGraph &CG);
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

/// \brief Get the inline cost for the always-inliner.
///
/// The always inliner *only* handles functions which are marked with the
/// attribute to force inlining. As such, it is dramatically simpler and avoids
/// using the powerful (but expensive) inline cost analysis. Instead it uses
/// a very simple and boring direct walk of the instructions looking for
/// impossible-to-inline constructs.
///
/// Note, it would be possible to go to some lengths to cache the information
/// computed here, but as we only expect to do this for relatively few and
/// small functions which have the explicit attribute to force inlining, it is
/// likely not worth it in practice.
InlineCost AlwaysInliner::getInlineCost(CallSite CS) {
  Function *Callee = CS.getCalledFunction();

  // Only inline direct calls to functions with always-inline attributes
  // that are viable for inlining. FIXME: We shouldn't even get here for
  // declarations.
  if (Callee && !Callee->isDeclaration() &&
      Callee->getFnAttributes().hasAttribute(Attribute::AlwaysInline) &&
      CA.isInlineViable(*Callee))
    return InlineCost::getAlways();

  return InlineCost::getNever();
}

// doInitialization - Initializes the vector of functions that have not
// been annotated with the "always inline" attribute.
bool AlwaysInliner::doInitialization(CallGraph &CG) {
  CA.setDataLayout(getAnalysisIfAvailable<DataLayout>());
  return false;
}
