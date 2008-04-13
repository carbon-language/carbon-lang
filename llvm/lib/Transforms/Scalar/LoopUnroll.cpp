//===-- LoopUnroll.cpp - Loop unroller pass -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements a simple loop unroller.  It works best when loops have
// been canonicalized by the -indvars pass, allowing it to determine the trip
// counts of loops easily.
//
// This pass will multi-block loops only if they contain no non-unrolled 
// subloops.  The process of unrolling can produce extraneous basic blocks 
// linked with unconditional branches.  This will be corrected in the future.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-unroll"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IntrinsicInst.h"
#include <algorithm>
#include <climits>
#include <cstdio>
using namespace llvm;

STATISTIC(NumCompletelyUnrolled, "Number of loops completely unrolled");
STATISTIC(NumUnrolled,    "Number of loops unrolled (completely or otherwise)");

namespace {
  cl::opt<unsigned>
  UnrollThreshold
    ("unroll-threshold", cl::init(100), cl::Hidden,
     cl::desc("The cut-off point for automatic loop unrolling"));

  cl::opt<unsigned>
  UnrollCount
    ("unroll-count", cl::init(0), cl::Hidden,
     cl::desc("Use this unroll count for all loops, for testing purposes"));

  class VISIBILITY_HIDDEN LoopUnroll : public LoopPass {
    LoopInfo *LI;  // The current loop information
  public:
    static char ID; // Pass ID, replacement for typeid
    LoopUnroll() : LoopPass((intptr_t)&ID) {}

    /// A magic value for use with the Threshold parameter to indicate
    /// that the loop unroll should be performed regardless of how much
    /// code expansion would result.
    static const unsigned NoThreshold = UINT_MAX;

    bool runOnLoop(Loop *L, LPPassManager &LPM);
    bool unrollLoop(Loop *L, unsigned Count, unsigned Threshold);
    BasicBlock *FoldBlockIntoPredecessor(BasicBlock *BB);

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG...
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      AU.addRequired<LoopInfo>();
      AU.addPreservedID(LCSSAID);
      AU.addPreserved<LoopInfo>();
    }
  };
  char LoopUnroll::ID = 0;
  RegisterPass<LoopUnroll> X("loop-unroll", "Unroll loops");
}

LoopPass *llvm::createLoopUnrollPass() { return new LoopUnroll(); }

/// ApproximateLoopSize - Approximate the size of the loop.
static unsigned ApproximateLoopSize(const Loop *L) {
  unsigned Size = 0;
  for (unsigned i = 0, e = L->getBlocks().size(); i != e; ++i) {
    BasicBlock *BB = L->getBlocks()[i];
    Instruction *Term = BB->getTerminator();
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      if (isa<PHINode>(I) && BB == L->getHeader()) {
        // Ignore PHI nodes in the header.
      } else if (I->hasOneUse() && I->use_back() == Term) {
        // Ignore instructions only used by the loop terminator.
      } else if (isa<DbgInfoIntrinsic>(I)) {
        // Ignore debug instructions
      } else if (isa<CallInst>(I)) {
        // Estimate size overhead introduced by call instructions which
        // is higher than other instructions. Here 3 and 10 are magic
        // numbers that help one isolated test case from PR2067 without
        // negatively impacting measured benchmarks.
        if (isa<IntrinsicInst>(I))
          Size = Size + 3;
        else
          Size = Size + 10;
      } else {
        ++Size;
      }

      // TODO: Ignore expressions derived from PHI and constants if inval of phi
      // is a constant, or if operation is associative.  This will get induction
      // variables.
    }
  }

  return Size;
}

// RemapInstruction - Convert the instruction operands from referencing the
// current values into those specified by ValueMap.
//
static inline void RemapInstruction(Instruction *I,
                                    DenseMap<const Value *, Value*> &ValueMap) {
  for (unsigned op = 0, E = I->getNumOperands(); op != E; ++op) {
    Value *Op = I->getOperand(op);
    DenseMap<const Value *, Value*>::iterator It = ValueMap.find(Op);
    if (It != ValueMap.end()) Op = It->second;
    I->setOperand(op, Op);
  }
}

// FoldBlockIntoPredecessor - Folds a basic block into its predecessor if it
// only has one predecessor, and that predecessor only has one successor.
// Returns the new combined block.
BasicBlock *LoopUnroll::FoldBlockIntoPredecessor(BasicBlock *BB) {
  // Merge basic blocks into their predecessor if there is only one distinct
  // pred, and if there is only one distinct successor of the predecessor, and
  // if there are no PHI nodes.
  //
  BasicBlock *OnlyPred = BB->getSinglePredecessor();
  if (!OnlyPred) return 0;

  if (OnlyPred->getTerminator()->getNumSuccessors() != 1)
    return 0;

  DOUT << "Merging: " << *BB << "into: " << *OnlyPred;

  // Resolve any PHI nodes at the start of the block.  They are all
  // guaranteed to have exactly one entry if they exist, unless there are
  // multiple duplicate (but guaranteed to be equal) entries for the
  // incoming edges.  This occurs when there are multiple edges from
  // OnlyPred to OnlySucc.
  //
  while (PHINode *PN = dyn_cast<PHINode>(&BB->front())) {
    PN->replaceAllUsesWith(PN->getIncomingValue(0));
    BB->getInstList().pop_front();  // Delete the phi node...
  }

  // Delete the unconditional branch from the predecessor...
  OnlyPred->getInstList().pop_back();

  // Move all definitions in the successor to the predecessor...
  OnlyPred->getInstList().splice(OnlyPred->end(), BB->getInstList());

  // Make all PHI nodes that referred to BB now refer to Pred as their
  // source...
  BB->replaceAllUsesWith(OnlyPred);

  std::string OldName = BB->getName();

  // Erase basic block from the function...
  LI->removeBlock(BB);
  BB->eraseFromParent();

  // Inherit predecessor's name if it exists...
  if (BB->hasName() && !OnlyPred->hasName())
    OnlyPred->takeName(BB);

  return OnlyPred;
}

bool LoopUnroll::runOnLoop(Loop *L, LPPassManager &LPM) {
  LI = &getAnalysis<LoopInfo>();

  // Unroll the loop.
  if (!unrollLoop(L, UnrollCount, UnrollThreshold))
    return false;

  // Update the loop information for this loop.
  // If we completely unrolled the loop, remove it from the parent.
  if (L->getNumBackEdges() == 0)
    LPM.deleteLoopFromQueue(L);

  return true;
}

/// Unroll the given loop by UnrollCount, or by a heuristically-determined
/// value if Count is zero. If Threshold is not NoThreshold, it is a value
/// to limit code size expansion. If the loop size would expand beyond the
/// threshold value, unrolling is suppressed. The return value is true if
/// any transformations are performed.
///
bool LoopUnroll::unrollLoop(Loop *L, unsigned Count, unsigned Threshold) {
  assert(L->isLCSSAForm());

  BasicBlock *Header = L->getHeader();
  BasicBlock *LatchBlock = L->getLoopLatch();
  BranchInst *BI = dyn_cast<BranchInst>(LatchBlock->getTerminator());

  DOUT << "Loop Unroll: F[" << Header->getParent()->getName()
       << "] Loop %" << Header->getName() << "\n";

  if (!BI || BI->isUnconditional()) {
    // The loop-rotate pass can be helpful to avoid this in many cases.
    DOUT << "  Can't unroll; loop not terminated by a conditional branch.\n";
    return false;
  }

  // Determine the trip count and/or trip multiple. A TripCount value of zero
  // is used to mean an unknown trip count. The TripMultiple value is the
  // greatest known integer multiple of the trip count.
  unsigned TripCount = 0;
  unsigned TripMultiple = 1;
  if (Value *TripCountValue = L->getTripCount()) {
    if (ConstantInt *TripCountC = dyn_cast<ConstantInt>(TripCountValue)) {
      // Guard against huge trip counts. This also guards against assertions in
      // APInt from the use of getZExtValue, below.
      if (TripCountC->getValue().getActiveBits() <= 32) {
        TripCount = (unsigned)TripCountC->getZExtValue();
      }
    } else if (BinaryOperator *BO = dyn_cast<BinaryOperator>(TripCountValue)) {
      switch (BO->getOpcode()) {
      case BinaryOperator::Mul:
        if (ConstantInt *MultipleC = dyn_cast<ConstantInt>(BO->getOperand(1))) {
          if (MultipleC->getValue().getActiveBits() <= 32) {
            TripMultiple = (unsigned)MultipleC->getZExtValue();
          }
        }
        break;
      default: break;
      }
    }
  }
  if (TripCount != 0)
    DOUT << "  Trip Count = " << TripCount << "\n";
  if (TripMultiple != 1)
    DOUT << "  Trip Multiple = " << TripMultiple << "\n";

  // Automatically select an unroll count.
  if (Count == 0) {
    // Conservative heuristic: if we know the trip count, see if we can
    // completely unroll (subject to the threshold, checked below); otherwise
    // don't unroll.
    if (TripCount != 0) {
      Count = TripCount;
    } else {
      return false;
    }
  }

  // Effectively "DCE" unrolled iterations that are beyond the tripcount
  // and will never be executed.
  if (TripCount != 0 && Count > TripCount)
    Count = TripCount;

  assert(Count > 0);
  assert(TripMultiple > 0);
  assert(TripCount == 0 || TripCount % TripMultiple == 0);

  // Enforce the threshold.
  if (Threshold != NoThreshold) {
    unsigned LoopSize = ApproximateLoopSize(L);
    DOUT << "  Loop Size = " << LoopSize << "\n";
    uint64_t Size = (uint64_t)LoopSize*Count;
    if (TripCount != 1 && Size > Threshold) {
      DOUT << "  TOO LARGE TO UNROLL: "
           << Size << ">" << Threshold << "\n";
      return false;
    }
  }

  // Are we eliminating the loop control altogether?
  bool CompletelyUnroll = Count == TripCount;

  // If we know the trip count, we know the multiple...
  unsigned BreakoutTrip = 0;
  if (TripCount != 0) {
    BreakoutTrip = TripCount % Count;
    TripMultiple = 0;
  } else {
    // Figure out what multiple to use.
    BreakoutTrip = TripMultiple =
      (unsigned)GreatestCommonDivisor64(Count, TripMultiple);
  }

  if (CompletelyUnroll) {
    DOUT << "COMPLETELY UNROLLING loop %" << Header->getName()
         << " with trip count " << TripCount << "!\n";
  } else {
    DOUT << "UNROLLING loop %" << Header->getName()
         << " by " << Count;
    if (TripMultiple == 0 || BreakoutTrip != TripMultiple) {
      DOUT << " with a breakout at trip " << BreakoutTrip;
    } else if (TripMultiple != 1) {
      DOUT << " with " << TripMultiple << " trips per branch";
    }
    DOUT << "!\n";
  }

  std::vector<BasicBlock*> LoopBlocks = L->getBlocks();

  bool ContinueOnTrue = L->contains(BI->getSuccessor(0));
  BasicBlock *LoopExit = BI->getSuccessor(ContinueOnTrue);

  // For the first iteration of the loop, we should use the precloned values for
  // PHI nodes.  Insert associations now.
  typedef DenseMap<const Value*, Value*> ValueMapTy;
  ValueMapTy LastValueMap;
  std::vector<PHINode*> OrigPHINode;
  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    OrigPHINode.push_back(PN);
    if (Instruction *I = 
                dyn_cast<Instruction>(PN->getIncomingValueForBlock(LatchBlock)))
      if (L->contains(I->getParent()))
        LastValueMap[I] = I;
  }

  std::vector<BasicBlock*> Headers;
  std::vector<BasicBlock*> Latches;
  Headers.push_back(Header);
  Latches.push_back(LatchBlock);

  for (unsigned It = 1; It != Count; ++It) {
    char SuffixBuffer[100];
    sprintf(SuffixBuffer, ".%d", It);
    
    std::vector<BasicBlock*> NewBlocks;
    
    for (std::vector<BasicBlock*>::iterator BB = LoopBlocks.begin(),
         E = LoopBlocks.end(); BB != E; ++BB) {
      ValueMapTy ValueMap;
      BasicBlock *New = CloneBasicBlock(*BB, ValueMap, SuffixBuffer);
      Header->getParent()->getBasicBlockList().push_back(New);

      // Loop over all of the PHI nodes in the block, changing them to use the
      // incoming values from the previous block.
      if (*BB == Header)
        for (unsigned i = 0, e = OrigPHINode.size(); i != e; ++i) {
          PHINode *NewPHI = cast<PHINode>(ValueMap[OrigPHINode[i]]);
          Value *InVal = NewPHI->getIncomingValueForBlock(LatchBlock);
          if (Instruction *InValI = dyn_cast<Instruction>(InVal))
            if (It > 1 && L->contains(InValI->getParent()))
              InVal = LastValueMap[InValI];
          ValueMap[OrigPHINode[i]] = InVal;
          New->getInstList().erase(NewPHI);
        }

      // Update our running map of newest clones
      LastValueMap[*BB] = New;
      for (ValueMapTy::iterator VI = ValueMap.begin(), VE = ValueMap.end();
           VI != VE; ++VI)
        LastValueMap[VI->first] = VI->second;

      L->addBasicBlockToLoop(New, LI->getBase());

      // Add phi entries for newly created values to all exit blocks except
      // the successor of the latch block.  The successor of the exit block will
      // be updated specially after unrolling all the way.
      if (*BB != LatchBlock)
        for (Value::use_iterator UI = (*BB)->use_begin(), UE = (*BB)->use_end();
             UI != UE;) {
          Instruction *UseInst = cast<Instruction>(*UI);
          ++UI;
          if (isa<PHINode>(UseInst) && !L->contains(UseInst->getParent())) {
            PHINode *phi = cast<PHINode>(UseInst);
            Value *Incoming = phi->getIncomingValueForBlock(*BB);
            phi->addIncoming(Incoming, New);
          }
        }

      // Keep track of new headers and latches as we create them, so that
      // we can insert the proper branches later.
      if (*BB == Header)
        Headers.push_back(New);
      if (*BB == LatchBlock) {
        Latches.push_back(New);

        // Also, clear out the new latch's back edge so that it doesn't look
        // like a new loop, so that it's amenable to being merged with adjacent
        // blocks later on.
        TerminatorInst *Term = New->getTerminator();
        assert(L->contains(Term->getSuccessor(!ContinueOnTrue)));
        assert(Term->getSuccessor(ContinueOnTrue) == LoopExit);
        Term->setSuccessor(!ContinueOnTrue, NULL);
      }

      NewBlocks.push_back(New);
    }
    
    // Remap all instructions in the most recent iteration
    for (unsigned i = 0; i < NewBlocks.size(); ++i) {
      BasicBlock *NB = NewBlocks[i];
      if (BasicBlock *UnwindDest = NB->getUnwindDest())
        NB->setUnwindDest(cast<BasicBlock>(LastValueMap[UnwindDest]));

      for (BasicBlock::iterator I = NB->begin(), E = NB->end(); I != E; ++I)
        RemapInstruction(I, LastValueMap);
    }
  }
  
  // The latch block exits the loop.  If there are any PHI nodes in the
  // successor blocks, update them to use the appropriate values computed as the
  // last iteration of the loop.
  if (Count != 1) {
    SmallPtrSet<PHINode*, 8> Users;
    for (Value::use_iterator UI = LatchBlock->use_begin(),
         UE = LatchBlock->use_end(); UI != UE; ++UI)
      if (PHINode *phi = dyn_cast<PHINode>(*UI))
        Users.insert(phi);
    
    BasicBlock *LastIterationBB = cast<BasicBlock>(LastValueMap[LatchBlock]);
    for (SmallPtrSet<PHINode*,8>::iterator SI = Users.begin(), SE = Users.end();
         SI != SE; ++SI) {
      PHINode *PN = *SI;
      Value *InVal = PN->removeIncomingValue(LatchBlock, false);
      // If this value was defined in the loop, take the value defined by the
      // last iteration of the loop.
      if (Instruction *InValI = dyn_cast<Instruction>(InVal)) {
        if (L->contains(InValI->getParent()))
          InVal = LastValueMap[InVal];
      }
      PN->addIncoming(InVal, LastIterationBB);
    }
  }

  // Now, if we're doing complete unrolling, loop over the PHI nodes in the
  // original block, setting them to their incoming values.
  if (CompletelyUnroll) {
    BasicBlock *Preheader = L->getLoopPreheader();
    for (unsigned i = 0, e = OrigPHINode.size(); i != e; ++i) {
      PHINode *PN = OrigPHINode[i];
      PN->replaceAllUsesWith(PN->getIncomingValueForBlock(Preheader));
      Header->getInstList().erase(PN);
    }
  }

  // Now that all the basic blocks for the unrolled iterations are in place,
  // set up the branches to connect them.
  for (unsigned i = 0, e = Latches.size(); i != e; ++i) {
    // The original branch was replicated in each unrolled iteration.
    BranchInst *Term = cast<BranchInst>(Latches[i]->getTerminator());

    // The branch destination.
    unsigned j = (i + 1) % e;
    BasicBlock *Dest = Headers[j];
    bool NeedConditional = true;

    // For a complete unroll, make the last iteration end with a branch
    // to the exit block.
    if (CompletelyUnroll && j == 0) {
      Dest = LoopExit;
      NeedConditional = false;
    }

    // If we know the trip count or a multiple of it, we can safely use an
    // unconditional branch for some iterations.
    if (j != BreakoutTrip && (TripMultiple == 0 || j % TripMultiple != 0)) {
      NeedConditional = false;
    }

    if (NeedConditional) {
      // Update the conditional branch's successor for the following
      // iteration.
      Term->setSuccessor(!ContinueOnTrue, Dest);
    } else {
      Term->setUnconditionalDest(Dest);
      // Merge adjacent basic blocks, if possible.
      if (BasicBlock *Fold = FoldBlockIntoPredecessor(Dest)) {
        std::replace(Latches.begin(), Latches.end(), Dest, Fold);
        std::replace(Headers.begin(), Headers.end(), Dest, Fold);
      }
    }
  }
  
  // At this point, the code is well formed.  We now do a quick sweep over the
  // inserted code, doing constant propagation and dead code elimination as we
  // go.
  const std::vector<BasicBlock*> &NewLoopBlocks = L->getBlocks();
  for (std::vector<BasicBlock*>::const_iterator BB = NewLoopBlocks.begin(),
       BBE = NewLoopBlocks.end(); BB != BBE; ++BB)
    for (BasicBlock::iterator I = (*BB)->begin(), E = (*BB)->end(); I != E; ) {
      Instruction *Inst = I++;

      if (isInstructionTriviallyDead(Inst))
        (*BB)->getInstList().erase(Inst);
      else if (Constant *C = ConstantFoldInstruction(Inst)) {
        Inst->replaceAllUsesWith(C);
        (*BB)->getInstList().erase(Inst);
      }
    }

  NumCompletelyUnrolled += CompletelyUnroll;
  ++NumUnrolled;
  return true;
}
