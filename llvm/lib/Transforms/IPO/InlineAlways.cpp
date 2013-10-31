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
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Transforms/IPO/InlinerPass.h"

using namespace llvm;

namespace {

/// \brief Inliner pass which only handles "always inline" functions.
class AlwaysInliner : public Inliner {
  InlineCostAnalysis *ICA;

public:
  // Use extremely low threshold.
  AlwaysInliner() : Inliner(ID, -2000000000, /*InsertLifetime*/ true), ICA(0) {
    initializeAlwaysInlinerPass(*PassRegistry::getPassRegistry());
  }

  AlwaysInliner(bool InsertLifetime)
      : Inliner(ID, -2000000000, InsertLifetime), ICA(0) {
    initializeAlwaysInlinerPass(*PassRegistry::getPassRegistry());
  }

  static char ID; // Pass identification, replacement for typeid

  virtual InlineCost getInlineCost(CallSite CS);

  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual bool runOnSCC(CallGraphSCC &SCC);

  using llvm::Pass::doFinalization;
  virtual bool doFinalization(CallGraph &CG) {
    return removeDeadFunctions(CG, /*AlwaysInlineOnly=*/ true);
  }
};

}

char AlwaysInliner::ID = 0;
INITIALIZE_PASS_BEGIN(AlwaysInliner, "always-inline",
                "Inliner for always_inline functions", false, false)
INITIALIZE_PASS_DEPENDENCY(CallGraph)
INITIALIZE_PASS_DEPENDENCY(InlineCostAnalysis)
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
      Callee->getAttributes().hasAttribute(AttributeSet::FunctionIndex,
                                           Attribute::AlwaysInline) &&
      ICA->isInlineViable(*Callee))
    return InlineCost::getAlways();

  return InlineCost::getNever();
}

bool AlwaysInliner::runOnSCC(CallGraphSCC &SCC) {
  ICA = &getAnalysis<InlineCostAnalysis>();
  return Inliner::runOnSCC(SCC);
}

void AlwaysInliner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<InlineCostAnalysis>();
  Inliner::getAnalysisUsage(AU);
}
