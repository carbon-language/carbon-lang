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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IntrinsicInst.h"
#include <cstdio>
#include <set>
#include <algorithm>
#include <iostream>
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
    BasicBlock* FoldBlockIntoPredecessor(BasicBlock* BB);

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
  RegisterPass<LoopUnroll> X("loop-unroll", "Unroll loops");
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
      } else if (DbgInfoIntrinsic *DbgI = dyn_cast<DbgInfoIntrinsic>(I)) {
        // Ignore debug instructions
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

// FoldBlockIntoPredecessor - Folds a basic block into its predecessor if it
// only has one predecessor, and that predecessor only has one successor.
// Returns the new combined block.
BasicBlock* LoopUnroll::FoldBlockIntoPredecessor(BasicBlock* BB) {
  // Merge basic blocks into their predecessor if there is only one distinct
  // pred, and if there is only one distinct successor of the predecessor, and
  // if there are no PHI nodes.
  //
  pred_iterator PI(pred_begin(BB)), PE(pred_end(BB));
  BasicBlock *OnlyPred = *PI++;
  for (; PI != PE; ++PI)  // Search all predecessors, see if they are all same
    if (*PI != OnlyPred) {
      OnlyPred = 0;       // There are multiple different predecessors...
      break;
    }

  BasicBlock *OnlySucc = 0;
  if (OnlyPred && OnlyPred != BB &&    // Don't break self loops
      OnlyPred->getTerminator()->getOpcode() != Instruction::Invoke) {
    // Check to see if there is only one distinct successor...
    succ_iterator SI(succ_begin(OnlyPred)), SE(succ_end(OnlyPred));
    OnlySucc = BB;
    for (; SI != SE; ++SI)
      if (*SI != OnlySucc) {
        OnlySucc = 0;     // There are multiple distinct successors!
        break;
      }
  }

  if (OnlySucc) {
    DEBUG(std::cerr << "Merging: " << *BB << "into: " << *OnlyPred);
    TerminatorInst *Term = OnlyPred->getTerminator();

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

    // Inherit predecessors name if it exists...
    if (!OldName.empty() && !OnlyPred->hasName())
      OnlyPred->setName(OldName);

    return OnlyPred;
  }
  
  return 0;
}

bool LoopUnroll::visitLoop(Loop *L) {
  bool Changed = false;

  // Recurse through all subloops before we process this loop.  Copy the loop
  // list so that the child can update the loop tree if it needs to delete the
  // loop.
  std::vector<Loop*> SubLoops(L->begin(), L->end());
  for (unsigned i = 0, e = SubLoops.size(); i != e; ++i)
    Changed |= visitLoop(SubLoops[i]);

  BasicBlock* Header = L->getHeader();
  BasicBlock* LatchBlock = L->getLoopLatch();

  BranchInst *BI = dyn_cast<BranchInst>(LatchBlock->getTerminator());
  if (BI == 0) return Changed;  // Must end in a conditional branch

  ConstantInt *TripCountC = dyn_cast_or_null<ConstantInt>(L->getTripCount());
  if (!TripCountC) return Changed;  // Must have constant trip count!

  uint64_t TripCountFull = TripCountC->getRawValue();
  if (TripCountFull != TripCountC->getRawValue() || TripCountFull == 0)
    return Changed; // More than 2^32 iterations???

  unsigned LoopSize = ApproximateLoopSize(L);
  DEBUG(std::cerr << "Loop Unroll: F[" << Header->getParent()->getName()
        << "] Loop %" << Header->getName() << " Loop Size = "
        << LoopSize << " Trip Count = " << TripCountFull << " - ");
  uint64_t Size = (uint64_t)LoopSize*TripCountFull;
  if (Size > UnrollThreshold) {
    DEBUG(std::cerr << "TOO LARGE: " << Size << ">" << UnrollThreshold << "\n");
    return Changed;
  }
  DEBUG(std::cerr << "UNROLLING!\n");

  std::vector<BasicBlock*> LoopBlocks = L->getBlocks();

  unsigned TripCount = (unsigned)TripCountFull;

  BasicBlock *LoopExit = BI->getSuccessor(L->contains(BI->getSuccessor(0))); 

  // For the first iteration of the loop, we should use the precloned values for
  // PHI nodes.  Insert associations now.
  std::map<const Value*, Value*> LastValueMap;
  std::vector<PHINode*> OrigPHINode;
  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    OrigPHINode.push_back(PN);
    if (Instruction *I = 
                dyn_cast<Instruction>(PN->getIncomingValueForBlock(LatchBlock)))
      if (L->contains(I->getParent()))
        LastValueMap[I] = I;
  }

  // Remove the exit branch from the loop
  LatchBlock->getInstList().erase(BI);
  
  std::vector<BasicBlock*> Headers;
  std::vector<BasicBlock*> Latches;
  Headers.push_back(Header);
  Latches.push_back(LatchBlock);

  assert(TripCount != 0 && "Trip count of 0 is impossible!");
  for (unsigned It = 1; It != TripCount; ++It) {
    char SuffixBuffer[100];
    sprintf(SuffixBuffer, ".%d", It);
    
    std::vector<BasicBlock*> NewBlocks;
    
    for (std::vector<BasicBlock*>::iterator BB = LoopBlocks.begin(),
         E = LoopBlocks.end(); BB != E; ++BB) {
      std::map<const Value*, Value*> ValueMap;
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
      for (std::map<const Value*, Value*>::iterator VI = ValueMap.begin(),
           VE = ValueMap.end(); VI != VE; ++VI)
        LastValueMap[VI->first] = VI->second;

      L->addBasicBlockToLoop(New, *LI);

      // Add phi entries for newly created values to all exit blocks except
      // the successor of the latch block.  The successor of the exit block will
      // be updated specially after unrolling all the way.
      if (*BB != LatchBlock)
        for (Value::use_iterator UI = (*BB)->use_begin(), UE = (*BB)->use_end();
             UI != UE; ++UI) {
          Instruction* UseInst = cast<Instruction>(*UI);
          if (isa<PHINode>(UseInst) && !L->contains(UseInst->getParent())) {
            PHINode* phi = cast<PHINode>(UseInst);
            Value* Incoming = phi->getIncomingValueForBlock(*BB);
            if (isa<Instruction>(Incoming))
              Incoming = LastValueMap[Incoming];
          
            phi->addIncoming(Incoming, New);
          }
        }

      // Keep track of new headers and latches as we create them, so that
      // we can insert the proper branches later.
      if (*BB == Header)
        Headers.push_back(New);
      if (*BB == LatchBlock)
        Latches.push_back(New);

      NewBlocks.push_back(New);
    }
    
    // Remap all instructions in the most recent iteration
    for (unsigned i = 0; i < NewBlocks.size(); ++i)
      for (BasicBlock::iterator I = NewBlocks[i]->begin(),
           E = NewBlocks[i]->end(); I != E; ++I)
        RemapInstruction(I, LastValueMap);
  }

  
 
  // Update PHI nodes that reference the final latch block
  if (TripCount > 1) {
    std::set<PHINode*> Users;
    for (Value::use_iterator UI = LatchBlock->use_begin(),
         UE = LatchBlock->use_end(); UI != UE; ++UI)
      if (PHINode* phi = dyn_cast<PHINode>(*UI))
        Users.insert(phi);
        
    for (std::set<PHINode*>::iterator SI = Users.begin(), SE = Users.end();
         SI != SE; ++SI) {
      Value* InVal = (*SI)->getIncomingValueForBlock(LatchBlock);
      if (isa<Instruction>(InVal))
        InVal = LastValueMap[InVal];
      (*SI)->removeIncomingValue(LatchBlock, false);
      if (InVal)
        (*SI)->addIncoming(InVal, cast<BasicBlock>(LastValueMap[LatchBlock]));
    }
  }

  // Now loop over the PHI nodes in the original block, setting them to their
  // incoming values.
  BasicBlock *Preheader = L->getLoopPreheader();
  for (unsigned i = 0, e = OrigPHINode.size(); i != e; ++i) {
    PHINode *PN = OrigPHINode[i];
    PN->replaceAllUsesWith(PN->getIncomingValueForBlock(Preheader));
    Header->getInstList().erase(PN);
  }
  
  //  Insert the branches that link the different iterations together
  for (unsigned i = 0; i < Latches.size()-1; ++i) {
    new BranchInst(Headers[i+1], Latches[i]);
    if(BasicBlock* Fold = FoldBlockIntoPredecessor(Headers[i+1])) {
      std::replace(Latches.begin(), Latches.end(), Headers[i+1], Fold);
      std::replace(Headers.begin(), Headers.end(), Headers[i+1], Fold);
    }
  }
  
  // Finally, add an unconditional branch to the block to continue into the exit
  // block.
  new BranchInst(LoopExit, Latches[Latches.size()-1]);
  FoldBlockIntoPredecessor(LoopExit);
  
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

  // Update the loop information for this loop.
  Loop *Parent = L->getParentLoop();

  // Move all of the basic blocks in the loop into the parent loop.
  for (std::vector<BasicBlock*>::const_iterator BB = NewLoopBlocks.begin(),
       E = NewLoopBlocks.end(); BB != E; ++BB)
    LI->changeLoopFor(*BB, Parent);

  // Remove the loop from the parent.
  if (Parent)
    delete Parent->removeChildLoop(std::find(Parent->begin(), Parent->end(),L));
  else
    delete LI->removeLoop(std::find(LI->begin(), LI->end(), L));

  ++NumUnrolled;
  return true;
}
