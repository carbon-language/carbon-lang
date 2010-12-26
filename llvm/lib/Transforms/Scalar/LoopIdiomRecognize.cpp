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
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

// TODO: Recognize "N" size array multiplies: replace with call to blas or
// something.

namespace {
  class LoopIdiomRecognize : public LoopPass {
    Loop *CurLoop;
    const TargetData *TD;
    ScalarEvolution *SE;
  public:
    static char ID;
    explicit LoopIdiomRecognize() : LoopPass(ID) {
      initializeLoopIdiomRecognizePass(*PassRegistry::getPassRegistry());
    }

    bool runOnLoop(Loop *L, LPPassManager &LPM);

    bool processLoopStore(StoreInst *SI, const SCEV *BECount);
    
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
  CurLoop = L;
  
  // We only look at trivial single basic block loops.
  // TODO: eventually support more complex loops, scanning the header.
  if (L->getBlocks().size() != 1)
    return false;
  
  // The trip count of the loop must be analyzable.
  SE = &getAnalysis<ScalarEvolution>();
  if (!SE->hasLoopInvariantBackedgeTakenCount(L))
    return false;
  const SCEV *BECount = SE->getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BECount)) return false;
  
  // We require target data for now.
  TD = getAnalysisIfAvailable<TargetData>();
  if (TD == 0) return false;
  
  BasicBlock *BB = L->getHeader();
  DEBUG(dbgs() << "loop-idiom Scanning: F[" << BB->getParent()->getName()
               << "] Loop %" << BB->getName() << "\n");

  bool MadeChange = false;
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
    // Look for store instructions, which may be memsets.
    if (StoreInst *SI = dyn_cast<StoreInst>(I++))
      MadeChange |= processLoopStore(SI, BECount);
  }
  
  return MadeChange;
}

/// scanBlock - Look over a block to see if we can promote anything out of it.
bool LoopIdiomRecognize::processLoopStore(StoreInst *SI, const SCEV *BECount) {
  Value *StoredVal = SI->getValueOperand();
  
  // Check to see if the store updates all bits in memory.  We don't want to
  // process things like a store of i3.  We also require that the store be a
  // multiple of a byte.
  uint64_t SizeInBits = TD->getTypeSizeInBits(StoredVal->getType());
  if ((SizeInBits & 7) || (SizeInBits >> 32) != 0 ||
      SizeInBits != TD->getTypeStoreSizeInBits(StoredVal->getType()))
    return false;
  
  // See if the pointer expression is an AddRec like {base,+,1} on the current
  // loop, which indicates a strided store.  If we have something else, it's a
  // random store we can't handle.
  const SCEVAddRecExpr *Ev =
    dyn_cast<SCEVAddRecExpr>(SE->getSCEV(SI->getPointerOperand()));
  if (Ev == 0 || Ev->getLoop() != CurLoop || !Ev->isAffine())
    return false;

  // Check to see if the stride matches the size of the store.  If so, then we
  // know that every byte is touched in the loop.
  unsigned StoreSize = (unsigned)SizeInBits >> 3; 
  const SCEVConstant *Stride = dyn_cast<SCEVConstant>(Ev->getOperand(1));
  if (Stride == 0 || StoreSize != Stride->getValue()->getValue())
    return false;
  
  errs() << "Found strided store: " << *Ev << "\n";
  
  // Check for memcpy here.
  
  
  // If the stored value is a byte-wise value (like i32 -1), then it may be
  // turned into a memset of i8 -1, assuming that all the consequtive bytes
  // are stored.  A store of i32 0x01020304 can never be turned into a memset.
  Value *SplatValue = isBytewiseValue(StoredVal);
  if (SplatValue == 0) return false;
  

  return false;
}

