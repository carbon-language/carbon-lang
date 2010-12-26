//===-- LoopIdiomRecognize.cpp - Loop idiom recognition -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements an idiom recognizer that transforms simple loops into a
// non-loop form.  In cases that this kicks in, it can be a significant
// performance win.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-idiom"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

// TODO: Recognize "N" size array multiplies: replace with call to blas or
// something.

namespace {
  class LoopIdiomRecognize : public LoopPass {
  public:
    static char ID;
    explicit LoopIdiomRecognize() : LoopPass(ID) {
      initializeLoopIdiomRecognizePass(*PassRegistry::getPassRegistry());
    }

    bool runOnLoop(Loop *L, LPPassManager &LPM);

    bool scanBlock(BasicBlock *BB, Loop *L);
    
    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG.
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
      AU.addRequired<ScalarEvolution>();
      AU.addPreserved<ScalarEvolution>();
      AU.addPreserved<DominatorTree>();
    }
  };
}

char LoopIdiomRecognize::ID = 0;
INITIALIZE_PASS_BEGIN(LoopIdiomRecognize, "loop-idiom", "Recognize loop idioms",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_END(LoopIdiomRecognize, "loop-idiom", "Recognize loop idioms",
                    false, false)

Pass *llvm::createLoopIdiomPass() { return new LoopIdiomRecognize(); }

bool LoopIdiomRecognize::runOnLoop(Loop *L, LPPassManager &LPM) {
  // We only look at trivial single basic block loops.
  // TODO: eventually support more complex loops, scanning the header.
  if (L->getBlocks().size() != 1)
    return false;
  
  BasicBlock *BB = L->getHeader();
  DEBUG(dbgs() << "Loop Idiom Recognize: F[" << BB->getParent()->getName()
               << "] Loop %" << BB->getName() << "\n");

  return scanBlock(BB, L);
}

/// scanBlock - Look over a block to see if we can promote anything out of it.
bool LoopIdiomRecognize::scanBlock(BasicBlock *BB, Loop *L) {
  ScalarEvolution &SE = getAnalysis<ScalarEvolution>();
  
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
    // Look for store instructions, which may be memsets.
    StoreInst *SI = dyn_cast<StoreInst>(I++);
    if (SI == 0) continue;
    
    // See if the pointer expression is an AddRec like {base,+,1} on the current
    // loop, which indicates a strided store.  If we have something else, it's a
    // random store we can't handle.
    const SCEVAddRecExpr *Ev =
      dyn_cast<SCEVAddRecExpr>(SE.getSCEV(SI->getPointerOperand()));
    if (Ev == 0 || Ev->getLoop() != L)
      continue;
    
    errs() << "Found strided store: " << *Ev << "\n";
  }
  
  return false;
}

