//===-- LoopUnroll.cpp - Loop unroller pass -------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass implements a simple loop unroller.  It works best when loops have
// been canonicalized by the -indvars pass, allowing it to determine the trip
// counts of loops easily.
//
// This pass is currently extremely limited.  It only currently only unrolls
// single basic block loops that execute a constant number of times.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-unroll"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include "Support/STLExtras.h"
#include <cstdio>
using namespace llvm;

namespace {
  Statistic<> NumUnrolled("loop-unroll", "Number of loops completely unrolled");

  cl::opt<unsigned>
  UnrollThreshold("unroll-threshold", cl::init(100), cl::Hidden,
                  cl::desc("The cut-off point for loop unrolling"));

  class LoopUnroll : public FunctionPass {
    LoopInfo *LI;  // The current loop information
  public:
    virtual bool runOnFunction(Function &F);
    bool visitLoop(Loop *L);

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG...
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
    }
  };
  RegisterOpt<LoopUnroll> X("loop-unroll", "Unroll loops");
}

FunctionPass *llvm::createLoopUnrollPass() { return new LoopUnroll(); }

bool LoopUnroll::runOnFunction(Function &F) {
  bool Changed = false;
  LI = &getAnalysis<LoopInfo>();

  // Transform all the top-level loops.  Copy the loop list so that the child
  // can update the loop tree if it needs to delete the loop.
  std::vector<Loop*> SubLoops(LI->begin(), LI->end());
  for (unsigned i = 0, e = SubLoops.size(); i != e; ++i)
    Changed |= visitLoop(SubLoops[i]);

  return Changed;
}

/// ApproximateLoopSize - Approximate the size of the loop after it has been
/// unrolled.
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
                                    std::map<const Value *, Value*> &ValueMap) {
  for (unsigned op = 0, E = I->getNumOperands(); op != E; ++op) {
    Value *Op = I->getOperand(op);
    std::map<const Value *, Value*>::iterator It = ValueMap.find(Op);
    if (It != ValueMap.end()) Op = It->second;
    I->setOperand(op, Op);
  }
}

static void ChangeExitBlocksFromTo(Loop::iterator I, Loop::iterator E,
                                   BasicBlock *Old, BasicBlock *New) {
  for (; I != E; ++I) {
    Loop *L = *I;
    if (L->hasExitBlock(Old)) {
      L->changeExitBlock(Old, New);
      ChangeExitBlocksFromTo(L->begin(), L->end(), Old, New);
    }
  }
}


bool LoopUnroll::visitLoop(Loop *L) {
  bool Changed = false;

  // Recurse through all subloops before we process this loop.  Copy the loop
  // list so that the child can update the loop tree if it needs to delete the
  // loop.
  std::vector<Loop*> SubLoops(L->begin(), L->end());
  for (unsigned i = 0, e = SubLoops.size(); i != e; ++i)
    Changed |= visitLoop(SubLoops[i]);

  // We only handle single basic block loops right now.
  if (L->getBlocks().size() != 1)
    return Changed;

  BasicBlock *BB = L->getHeader();
  BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator());
  if (BI == 0) return Changed;  // Must end in a conditional branch

  ConstantInt *TripCountC = dyn_cast_or_null<ConstantInt>(L->getTripCount());
  if (!TripCountC) return Changed;  // Must have constant trip count!

  unsigned TripCount = TripCountC->getRawValue();
  if (TripCount != TripCountC->getRawValue())
    return Changed; // More than 2^32 iterations???

  unsigned LoopSize = ApproximateLoopSize(L);
  DEBUG(std::cerr << "Loop Unroll: F[" << BB->getParent()->getName()
        << "] Loop %" << BB->getName() << " Loop Size = " << LoopSize
        << " Trip Count = " << TripCount << " - ");
  if (LoopSize*TripCount > UnrollThreshold) {
    DEBUG(std::cerr << "TOO LARGE: " << LoopSize*TripCount << ">"
                    << UnrollThreshold << "\n");
    return Changed;
  }
  DEBUG(std::cerr << "UNROLLING!\n");
  
  assert(L->getExitBlocks().size() == 1 && "Must have exactly one exit block!");
  BasicBlock *LoopExit = L->getExitBlocks()[0];

  // Create a new basic block to temporarily hold all of the cloned code.
  BasicBlock *NewBlock = new BasicBlock();

  // For the first iteration of the loop, we should use the precloned values for
  // PHI nodes.  Insert associations now.
  std::map<const Value*, Value*> LastValueMap;
  std::vector<PHINode*> OrigPHINode;
  for (BasicBlock::iterator I = BB->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I) {
    OrigPHINode.push_back(PN);
    if (Instruction *I =dyn_cast<Instruction>(PN->getIncomingValueForBlock(BB)))
      if (I->getParent() == BB)
        LastValueMap[I] = I;
  }

  // Remove the exit branch from the loop
  BB->getInstList().erase(BI);

  assert(TripCount != 0 && "Trip count of 0 is impossible!");
  for (unsigned It = 1; It != TripCount; ++It) {
    char SuffixBuffer[100];
    sprintf(SuffixBuffer, ".%d", It);
    std::map<const Value*, Value*> ValueMap;
    BasicBlock *New = CloneBasicBlock(BB, ValueMap, SuffixBuffer);

    // Loop over all of the PHI nodes in the block, changing them to use the
    // incoming values from the previous block.
    for (unsigned i = 0, e = OrigPHINode.size(); i != e; ++i) {
      PHINode *NewPHI = cast<PHINode>(ValueMap[OrigPHINode[i]]);
      Value *InVal = NewPHI->getIncomingValueForBlock(BB);
      if (Instruction *InValI = dyn_cast<Instruction>(InVal))
        if (InValI->getParent() == BB)
          InVal = LastValueMap[InValI];
      ValueMap[OrigPHINode[i]] = InVal;
      New->getInstList().erase(NewPHI);
    }

    for (BasicBlock::iterator I = New->begin(), E = New->end(); I != E; ++I)
      RemapInstruction(I, ValueMap);

    // Now that all of the instructions are remapped, splice them into the end
    // of the NewBlock.
    NewBlock->getInstList().splice(NewBlock->end(), New->getInstList());
    delete New;

    // LastValue map now contains values from this iteration.
    std::swap(LastValueMap, ValueMap);
  }

  // If there was more than one iteration, replace any uses of values computed
  // in the loop with values computed during the last iteration of the loop.
  if (TripCount != 1) {
    std::set<User*> Users;
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      Users.insert(I->use_begin(), I->use_end());

    // We don't want to reprocess entries with PHI nodes in them.  For this
    // reason, we look at each operand of each user exactly once, performing the
    // stubstitution exactly once.
    for (std::set<User*>::iterator UI = Users.begin(), E = Users.end(); UI != E;
         ++UI) {
      Instruction *I = cast<Instruction>(*UI);
      if (I->getParent() != BB && I->getParent() != NewBlock)
        RemapInstruction(I, LastValueMap);
    }
  }

  // Now that we cloned the block as many times as we needed, stitch the new
  // code into the original block and delete the temporary block.
  BB->getInstList().splice(BB->end(), NewBlock->getInstList());
  delete NewBlock;

  // Now loop over the PHI nodes in the original block, setting them to their
  // incoming values.
  BasicBlock *Preheader = L->getLoopPreheader();
  for (unsigned i = 0, e = OrigPHINode.size(); i != e; ++i) {
    PHINode *PN = OrigPHINode[i];
    PN->replaceAllUsesWith(PN->getIncomingValueForBlock(Preheader));
    BB->getInstList().erase(PN);
  }
 
  // Finally, add an unconditional branch to the block to continue into the exit
  // block.
  new BranchInst(LoopExit, BB);

  // At this point, the code is well formed.  We now do a quick sweep over the
  // inserted code, doing constant propagation and dead code elimination as we
  // go.
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
    Instruction *Inst = I++;
    
    if (isInstructionTriviallyDead(Inst))
      BB->getInstList().erase(Inst);
    else if (Constant *C = ConstantFoldInstruction(Inst)) {
      Inst->replaceAllUsesWith(C);
      BB->getInstList().erase(Inst);
    }
  }

  // Update the loop information for this loop.
  Loop *Parent = L->getParentLoop();

  // Move all of the basic blocks in the loop into the parent loop.
  LI->changeLoopFor(BB, Parent);

  // Remove the loop from the parent.
  if (Parent)
    delete Parent->removeChildLoop(std::find(Parent->begin(), Parent->end(),L));
  else
    delete LI->removeLoop(std::find(LI->begin(), LI->end(), L));


  // FIXME: Should update dominator analyses


  // Now that everything is up-to-date that will be, we fold the loop block into
  // the preheader and exit block, updating our analyses as we go.
  LoopExit->getInstList().splice(LoopExit->begin(), BB->getInstList(),
                                 BB->getInstList().begin(),
                                 prior(BB->getInstList().end()));
  LoopExit->getInstList().splice(LoopExit->begin(), Preheader->getInstList(),
                                 Preheader->getInstList().begin(),
                                 prior(Preheader->getInstList().end()));

  // Make all other blocks in the program branch to LoopExit now instead of
  // Preheader.
  Preheader->replaceAllUsesWith(LoopExit);

  // Remove BB and LoopExit from our analyses.
  LI->removeBlock(Preheader);
  LI->removeBlock(BB);

  // If any loops used Preheader as an exit block, update them to use LoopExit.
  if (Parent)
    ChangeExitBlocksFromTo(Parent->begin(), Parent->end(),
                           Preheader, LoopExit);
  else
    ChangeExitBlocksFromTo(LI->begin(), LI->end(),
                           Preheader, LoopExit);

  // If the preheader was the entry block of this function, move the exit block
  // to be the new entry of the loop.
  Function *F = LoopExit->getParent();
  if (Preheader == &F->front())
    F->getBasicBlockList().splice(F->begin(), F->getBasicBlockList(), LoopExit);

  // Actually delete the blocks now.
  F->getBasicBlockList().erase(Preheader);
  F->getBasicBlockList().erase(BB);

  ++NumUnrolled;
  return true;
}
