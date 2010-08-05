//===-- Sink.cpp - Code Sinking -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass moves instructions into successor blocks, when possible, so that
// they aren't executed on paths where their results aren't needed.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sink"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

STATISTIC(NumSunk, "Number of instructions sunk");

namespace {
  class Sinking : public FunctionPass {
    DominatorTree *DT;
    LoopInfo *LI;
    AliasAnalysis *AA;

  public:
    static char ID; // Pass identification
    Sinking() : FunctionPass(ID) {}
    
    virtual bool runOnFunction(Function &F);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      FunctionPass::getAnalysisUsage(AU);
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<DominatorTree>();
      AU.addRequired<LoopInfo>();
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<LoopInfo>();
    }
  private:
    bool ProcessBlock(BasicBlock &BB);
    bool SinkInstruction(Instruction *I, SmallPtrSet<Instruction *, 8> &Stores);
    bool AllUsesDominatedByBlock(Instruction *Inst, BasicBlock *BB) const;
  };
} // end anonymous namespace
  
char Sinking::ID = 0;
INITIALIZE_PASS(Sinking, "sink", "Code sinking", false, false);

FunctionPass *llvm::createSinkingPass() { return new Sinking(); }

/// AllUsesDominatedByBlock - Return true if all uses of the specified value
/// occur in blocks dominated by the specified block.
bool Sinking::AllUsesDominatedByBlock(Instruction *Inst, 
                                      BasicBlock *BB) const {
  // Ignoring debug uses is necessary so debug info doesn't affect the code.
  // This may leave a referencing dbg_value in the original block, before
  // the definition of the vreg.  Dwarf generator handles this although the
  // user might not get the right info at runtime.
  for (Value::use_iterator I = Inst->use_begin(),
       E = Inst->use_end(); I != E; ++I) {
    // Determine the block of the use.
    Instruction *UseInst = cast<Instruction>(*I);
    BasicBlock *UseBlock = UseInst->getParent();
    if (PHINode *PN = dyn_cast<PHINode>(UseInst)) {
      // PHI nodes use the operand in the predecessor block, not the block with
      // the PHI.
      unsigned Num = PHINode::getIncomingValueNumForOperand(I.getOperandNo());
      UseBlock = PN->getIncomingBlock(Num);
    }
    // Check that it dominates.
    if (!DT->dominates(BB, UseBlock))
      return false;
  }
  return true;
}

bool Sinking::runOnFunction(Function &F) {
  DT = &getAnalysis<DominatorTree>();
  LI = &getAnalysis<LoopInfo>();
  AA = &getAnalysis<AliasAnalysis>();

  bool EverMadeChange = false;
  
  while (1) {
    bool MadeChange = false;

    // Process all basic blocks.
    for (Function::iterator I = F.begin(), E = F.end(); 
         I != E; ++I)
      MadeChange |= ProcessBlock(*I);
    
    // If this iteration over the code changed anything, keep iterating.
    if (!MadeChange) break;
    EverMadeChange = true;
  } 
  return EverMadeChange;
}

bool Sinking::ProcessBlock(BasicBlock &BB) {
  // Can't sink anything out of a block that has less than two successors.
  if (BB.getTerminator()->getNumSuccessors() <= 1 || BB.empty()) return false;

  // Don't bother sinking code out of unreachable blocks. In addition to being
  // unprofitable, it can also lead to infinite looping, because in an unreachable
  // loop there may be nowhere to stop.
  if (!DT->isReachableFromEntry(&BB)) return false;

  bool MadeChange = false;

  // Walk the basic block bottom-up.  Remember if we saw a store.
  BasicBlock::iterator I = BB.end();
  --I;
  bool ProcessedBegin = false;
  SmallPtrSet<Instruction *, 8> Stores;
  do {
    Instruction *Inst = I;  // The instruction to sink.
    
    // Predecrement I (if it's not begin) so that it isn't invalidated by
    // sinking.
    ProcessedBegin = I == BB.begin();
    if (!ProcessedBegin)
      --I;

    if (isa<DbgInfoIntrinsic>(Inst))
      continue;

    if (SinkInstruction(Inst, Stores))
      ++NumSunk, MadeChange = true;
    
    // If we just processed the first instruction in the block, we're done.
  } while (!ProcessedBegin);
  
  return MadeChange;
}

static bool isSafeToMove(Instruction *Inst, AliasAnalysis *AA,
                         SmallPtrSet<Instruction *, 8> &Stores) {
  if (LoadInst *L = dyn_cast<LoadInst>(Inst)) {
    if (L->isVolatile()) return false;

    Value *Ptr = L->getPointerOperand();
    unsigned Size = AA->getTypeStoreSize(L->getType());
    for (SmallPtrSet<Instruction *, 8>::iterator I = Stores.begin(),
         E = Stores.end(); I != E; ++I)
      if (AA->getModRefInfo(*I, Ptr, Size) & AliasAnalysis::Mod)
        return false;
  }

  if (Inst->mayWriteToMemory()) {
    Stores.insert(Inst);
    return false;
  }

  return Inst->isSafeToSpeculativelyExecute();
}

/// SinkInstruction - Determine whether it is safe to sink the specified machine
/// instruction out of its current block into a successor.
bool Sinking::SinkInstruction(Instruction *Inst,
                              SmallPtrSet<Instruction *, 8> &Stores) {
  // Check if it's safe to move the instruction.
  if (!isSafeToMove(Inst, AA, Stores))
    return false;
  
  // FIXME: This should include support for sinking instructions within the
  // block they are currently in to shorten the live ranges.  We often get
  // instructions sunk into the top of a large block, but it would be better to
  // also sink them down before their first use in the block.  This xform has to
  // be careful not to *increase* register pressure though, e.g. sinking
  // "x = y + z" down if it kills y and z would increase the live ranges of y
  // and z and only shrink the live range of x.
  
  // Loop over all the operands of the specified instruction.  If there is
  // anything we can't handle, bail out.
  BasicBlock *ParentBlock = Inst->getParent();
  
  // SuccToSinkTo - This is the successor to sink this instruction to, once we
  // decide.
  BasicBlock *SuccToSinkTo = 0;
  
  // FIXME: This picks a successor to sink into based on having one
  // successor that dominates all the uses.  However, there are cases where
  // sinking can happen but where the sink point isn't a successor.  For
  // example:
  //   x = computation
  //   if () {} else {}
  //   use x
  // the instruction could be sunk over the whole diamond for the 
  // if/then/else (or loop, etc), allowing it to be sunk into other blocks
  // after that.
  
  // Instructions can only be sunk if all their uses are in blocks
  // dominated by one of the successors.
  // Look at all the successors and decide which one
  // we should sink to.
  for (succ_iterator SI = succ_begin(ParentBlock),
       E = succ_end(ParentBlock); SI != E; ++SI) {
    if (AllUsesDominatedByBlock(Inst, *SI)) {
      SuccToSinkTo = *SI;
      break;
    }
  }
      
  // If we couldn't find a block to sink to, ignore this instruction.
  if (SuccToSinkTo == 0)
    return false;
  
  // It is not possible to sink an instruction into its own block.  This can
  // happen with loops.
  if (Inst->getParent() == SuccToSinkTo)
    return false;
  
  DEBUG(dbgs() << "Sink instr " << *Inst);
  DEBUG(dbgs() << "to block ";
        WriteAsOperand(dbgs(), SuccToSinkTo, false));
  
  // If the block has multiple predecessors, this would introduce computation on
  // a path that it doesn't already exist.  We could split the critical edge,
  // but for now we just punt.
  // FIXME: Split critical edges if not backedges.
  if (SuccToSinkTo->getUniquePredecessor() != ParentBlock) {
    // We cannot sink a load across a critical edge - there may be stores in
    // other code paths.
    if (!Inst->isSafeToSpeculativelyExecute()) {
      DEBUG(dbgs() << " *** PUNTING: Wont sink load along critical edge.\n");
      return false;
    }

    // We don't want to sink across a critical edge if we don't dominate the
    // successor. We could be introducing calculations to new code paths.
    if (!DT->dominates(ParentBlock, SuccToSinkTo)) {
      DEBUG(dbgs() << " *** PUNTING: Critical edge found\n");
      return false;
    }

    // Don't sink instructions into a loop.
    if (LI->isLoopHeader(SuccToSinkTo)) {
      DEBUG(dbgs() << " *** PUNTING: Loop header found\n");
      return false;
    }

    // Otherwise we are OK with sinking along a critical edge.
    DEBUG(dbgs() << "Sinking along critical edge.\n");
  }
  
  // Determine where to insert into.  Skip phi nodes.
  BasicBlock::iterator InsertPos = SuccToSinkTo->begin();
  while (InsertPos != SuccToSinkTo->end() && isa<PHINode>(InsertPos))
    ++InsertPos;
  
  // Move the instruction.
  Inst->moveBefore(InsertPos);
  return true;
}
