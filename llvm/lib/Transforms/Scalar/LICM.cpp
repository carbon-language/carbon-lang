//===-- LICM.cpp - Loop Invariant Code Motion Pass ------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass performs loop invariant code motion, attempting to remove as much
// code from the body of a loop as possible.  It does this by either hoisting
// code into the preheader block, or by sinking code to the exit blocks if it is
// safe.  This pass also promotes must-aliased memory locations in the loop to
// live in registers.
//
// This pass uses alias analysis for two purposes:
//
//  1. Moving loop invariant loads out of loops.  If we can determine that a
//     load inside of a loop never aliases anything stored to, we can hoist it
//     or sink it like any other instruction.
//  2. Scalar Promotion of Memory - If there is a store instruction inside of
//     the loop, we try to move the store to happen AFTER the loop instead of
//     inside of the loop.  This can only happen if a few conditions are true:
//       A. The pointer stored through is loop invariant
//       B. There are no stores or loads in the loop which _may_ alias the
//          pointer.  There are no calls in the loop which mod/ref the pointer.
//     If these conditions are true, we can promote the loads and stores in the
//     loop of the pointer to use a temporary alloca'd variable.  We then use
//     the mem2reg functionality to construct the appropriate SSA form for the
//     variable.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Instructions.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CFG.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include "llvm/Assembly/Writer.h"
#include <algorithm>
using namespace llvm;

namespace {
  cl::opt<bool>
  DisablePromotion("disable-licm-promotion", cl::Hidden,
                   cl::desc("Disable memory promotion in LICM pass"));

  Statistic<> NumSunk("licm", "Number of instructions sunk out of loop");
  Statistic<> NumHoisted("licm", "Number of instructions hoisted out of loop");
  Statistic<> NumMovedLoads("licm", "Number of load insts hoisted or sunk");
  Statistic<> NumPromoted("licm",
                          "Number of memory locations promoted to registers");

  struct LICM : public FunctionPass {
    virtual bool runOnFunction(Function &F);

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG...
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequired<LoopInfo>();
      AU.addRequired<DominatorTree>();
      AU.addRequired<DominanceFrontier>();  // For scalar promotion (mem2reg)
      AU.addRequired<AliasAnalysis>();
    }

  private:
    // Various analyses that we use...
    AliasAnalysis *AA;       // Current AliasAnalysis information
    LoopInfo      *LI;       // Current LoopInfo
    DominatorTree *DT;       // Dominator Tree for the current Loop...
    DominanceFrontier *DF;   // Current Dominance Frontier

    // State that is updated as we process loops
    bool Changed;            // Set to true when we change anything.
    BasicBlock *Preheader;   // The preheader block of the current loop...
    Loop *CurLoop;           // The current loop we are working on...
    AliasSetTracker *CurAST; // AliasSet information for the current loop...

    /// visitLoop - Hoist expressions out of the specified loop...    
    ///
    void visitLoop(Loop *L, AliasSetTracker &AST);

    /// HoistRegion - Walk the specified region of the CFG (defined by all
    /// blocks dominated by the specified block, and that are in the current
    /// loop) in depth first order w.r.t the DominatorTree.  This allows us to
    /// visit definitions before uses, allowing us to hoist a loop body in one
    /// pass without iteration.
    ///
    void HoistRegion(DominatorTree::Node *N);

    /// inSubLoop - Little predicate that returns true if the specified basic
    /// block is in a subloop of the current one, not the current one itself.
    ///
    bool inSubLoop(BasicBlock *BB) {
      assert(CurLoop->contains(BB) && "Only valid if BB is IN the loop");
      for (unsigned i = 0, e = CurLoop->getSubLoops().size(); i != e; ++i)
        if (CurLoop->getSubLoops()[i]->contains(BB))
          return true;  // A subloop actually contains this block!
      return false;
    }

    /// isExitBlockDominatedByBlockInLoop - This method checks to see if the
    /// specified exit block of the loop is dominated by the specified block
    /// that is in the body of the loop.  We use these constraints to
    /// dramatically limit the amount of the dominator tree that needs to be
    /// searched.
    bool isExitBlockDominatedByBlockInLoop(BasicBlock *ExitBlock,
                                           BasicBlock *BlockInLoop) const {
      // If the block in the loop is the loop header, it must be dominated!
      BasicBlock *LoopHeader = CurLoop->getHeader();
      if (BlockInLoop == LoopHeader)
        return true;
      
      DominatorTree::Node *BlockInLoopNode = DT->getNode(BlockInLoop);
      DominatorTree::Node *IDom            = DT->getNode(ExitBlock);
    
      // Because the exit block is not in the loop, we know we have to get _at
      // least_ it's immediate dominator.
      do {
        // Get next Immediate Dominator.
        IDom = IDom->getIDom();
        
        // If we have got to the header of the loop, then the instructions block
        // did not dominate the exit node, so we can't hoist it.
        if (IDom->getBlock() == LoopHeader)
          return false;
        
      } while (IDom != BlockInLoopNode);

      return true;
    }

    /// sink - When an instruction is found to only be used outside of the loop,
    /// this function moves it to the exit blocks and patches up SSA form as
    /// needed.
    ///
    void sink(Instruction &I);

    /// hoist - When an instruction is found to only use loop invariant operands
    /// that is safe to hoist, this instruction is called to do the dirty work.
    ///
    void hoist(Instruction &I);

    /// isSafeToExecuteUnconditionally - Only sink or hoist an instruction if it
    /// is not a trapping instruction or if it is a trapping instruction and is
    /// guaranteed to execute.
    ///
    bool isSafeToExecuteUnconditionally(Instruction &I);

    /// pointerInvalidatedByLoop - Return true if the body of this loop may
    /// store into the memory location pointed to by V.
    /// 
    bool pointerInvalidatedByLoop(Value *V) {
      // Check to see if any of the basic blocks in CurLoop invalidate *V.
      return CurAST->getAliasSetForPointer(V, 0).isMod();
    }

    /// isLoopInvariant - Return true if the specified value is loop invariant
    ///
    inline bool isLoopInvariant(Value *V) {
      if (Instruction *I = dyn_cast<Instruction>(V))
        return !CurLoop->contains(I->getParent());
      return true;  // All non-instructions are loop invariant
    }

    bool canSinkOrHoistInst(Instruction &I);
    bool isLoopInvariantInst(Instruction &I);
    bool isNotUsedInLoop(Instruction &I);

    /// PromoteValuesInLoop - Look at the stores in the loop and promote as many
    /// to scalars as we can.
    ///
    void PromoteValuesInLoop();

    /// findPromotableValuesInLoop - Check the current loop for stores to
    /// definite pointers, which are not loaded and stored through may aliases.
    /// If these are found, create an alloca for the value, add it to the
    /// PromotedValues list, and keep track of the mapping from value to
    /// alloca...
    ///
    void findPromotableValuesInLoop(
                   std::vector<std::pair<AllocaInst*, Value*> > &PromotedValues,
                                    std::map<Value*, AllocaInst*> &Val2AlMap);
  };

  RegisterOpt<LICM> X("licm", "Loop Invariant Code Motion");
}

FunctionPass *llvm::createLICMPass() { return new LICM(); }

/// runOnFunction - For LICM, this simply traverses the loop structure of the
/// function, hoisting expressions out of loops if possible.
///
bool LICM::runOnFunction(Function &) {
  Changed = false;

  // Get our Loop and Alias Analysis information...
  LI = &getAnalysis<LoopInfo>();
  AA = &getAnalysis<AliasAnalysis>();
  DF = &getAnalysis<DominanceFrontier>();
  DT = &getAnalysis<DominatorTree>();

  // Hoist expressions out of all of the top-level loops.
  const std::vector<Loop*> &TopLevelLoops = LI->getTopLevelLoops();
  for (std::vector<Loop*>::const_iterator I = TopLevelLoops.begin(),
         E = TopLevelLoops.end(); I != E; ++I) {
    AliasSetTracker AST(*AA);
    visitLoop(*I, AST);
  }
  return Changed;
}


/// visitLoop - Hoist expressions out of the specified loop...    
///
void LICM::visitLoop(Loop *L, AliasSetTracker &AST) {
  // Recurse through all subloops before we process this loop...
  for (std::vector<Loop*>::const_iterator I = L->getSubLoops().begin(),
         E = L->getSubLoops().end(); I != E; ++I) {
    AliasSetTracker SubAST(*AA);
    visitLoop(*I, SubAST);

    // Incorporate information about the subloops into this loop...
    AST.add(SubAST);
  }
  CurLoop = L;
  CurAST = &AST;

  // Get the preheader block to move instructions into...
  Preheader = L->getLoopPreheader();
  assert(Preheader&&"Preheader insertion pass guarantees we have a preheader!");

  // Loop over the body of this loop, looking for calls, invokes, and stores.
  // Because subloops have already been incorporated into AST, we skip blocks in
  // subloops.
  //
  for (std::vector<BasicBlock*>::const_iterator I = L->getBlocks().begin(),
         E = L->getBlocks().end(); I != E; ++I)
    if (LI->getLoopFor(*I) == L)        // Ignore blocks in subloops...
      AST.add(**I);                     // Incorporate the specified basic block

  // We want to visit all of the instructions in this loop... that are not parts
  // of our subloops (they have already had their invariants hoisted out of
  // their loop, into this loop, so there is no need to process the BODIES of
  // the subloops).
  //
  // Traverse the body of the loop in depth first order on the dominator tree so
  // that we are guaranteed to see definitions before we see uses.  This allows
  // us to perform the LICM transformation in one pass, without iteration.
  //
  HoistRegion(DT->getNode(L->getHeader()));

  // Now that all loop invariants have been removed from the loop, promote any
  // memory references to scalars that we can...
  if (!DisablePromotion)
    PromoteValuesInLoop();

  // Clear out loops state information for the next iteration
  CurLoop = 0;
  Preheader = 0;
}

/// HoistRegion - Walk the specified region of the CFG (defined by all blocks
/// dominated by the specified block, and that are in the current loop) in depth
/// first order w.r.t the DominatorTree.  This allows us to visit definitions
/// before uses, allowing us to hoist a loop body in one pass without iteration.
///
void LICM::HoistRegion(DominatorTree::Node *N) {
  assert(N != 0 && "Null dominator tree node?");
  BasicBlock *BB = N->getBlock();

  // If this subregion is not in the top level loop at all, exit.
  if (!CurLoop->contains(BB)) return;

  // Only need to process the contents of this block if it is not part of a
  // subloop (which would already have been processed).
  if (!inSubLoop(BB))
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E; ) {
      Instruction &I = *II++;

      // We can only handle simple expressions and loads with this code.
      if (canSinkOrHoistInst(I)) {
        // First check to see if we can sink this instruction to the exit blocks
        // of the loop.  We can do this if the only users of the instruction are
        // outside of the loop.  In this case, it doesn't even matter if the
        // operands of the instruction are loop invariant.
        //
        if (isNotUsedInLoop(I))
          sink(I);
        
        // If we can't sink the instruction, try hoisting it out to the
        // preheader.  We can only do this if all of the operands of the
        // instruction are loop invariant and if it is safe to hoist the
        // instruction.
        //
        else if (isLoopInvariantInst(I) && isSafeToExecuteUnconditionally(I))
          hoist(I);
      }
    }

  const std::vector<DominatorTree::Node*> &Children = N->getChildren();
  for (unsigned i = 0, e = Children.size(); i != e; ++i)
    HoistRegion(Children[i]);
}

/// canSinkOrHoistInst - Return true if the hoister and sinker can handle this
/// instruction.
///
bool LICM::canSinkOrHoistInst(Instruction &I) {
  // Loads have extra constraints we have to verify before we can hoist them.
  if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
    if (LI->isVolatile())
      return false;        // Don't hoist volatile loads!

    // Don't hoist loads which have may-aliased stores in loop.
    return !pointerInvalidatedByLoop(LI->getOperand(0));
  }

  return isa<BinaryOperator>(I) || isa<ShiftInst>(I) || isa<CastInst>(I) || 
         isa<GetElementPtrInst>(I) || isa<VANextInst>(I) || isa<VAArgInst>(I);
}

/// isNotUsedInLoop - Return true if the only users of this instruction are
/// outside of the loop.  If this is true, we can sink the instruction to the
/// exit blocks of the loop.
///
bool LICM::isNotUsedInLoop(Instruction &I) {
  for (Value::use_iterator UI = I.use_begin(), E = I.use_end(); UI != E; ++UI)
    if (CurLoop->contains(cast<Instruction>(*UI)->getParent()))
      return false;
  return true;
}


/// isLoopInvariantInst - Return true if all operands of this instruction are
/// loop invariant.  We also filter out non-hoistable instructions here just for
/// efficiency.
///
bool LICM::isLoopInvariantInst(Instruction &I) {
  // The instruction is loop invariant if all of its operands are loop-invariant
  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i)
    if (!isLoopInvariant(I.getOperand(i)))
      return false;

  // If we got this far, the instruction is loop invariant!
  return true;
}

/// sink - When an instruction is found to only be used outside of the loop,
/// this function moves it to the exit blocks and patches up SSA form as
/// needed.
///
void LICM::sink(Instruction &I) {
  DEBUG(std::cerr << "LICM sinking instruction: " << I);

  const std::vector<BasicBlock*> &ExitBlocks = CurLoop->getExitBlocks();
  
  // The case where there is only a single exit node of this loop is common
  // enough that we handle it as a special (more efficient) case.  It is more
  // efficient to handle because there are no PHI nodes that need to be placed.
  if (ExitBlocks.size() == 1) {
    if (!isExitBlockDominatedByBlockInLoop(ExitBlocks[0], I.getParent())) {
      // Instruction is not used, just delete it.
      I.getParent()->getInstList().erase(&I);
    } else {
      // Move the instruction to the start of the exit block, after any PHI
      // nodes in it.
      I.getParent()->getInstList().remove(&I);
      
      BasicBlock::iterator InsertPt = ExitBlocks[0]->begin();
      while (isa<PHINode>(InsertPt)) ++InsertPt;
      ExitBlocks[0]->getInstList().insert(InsertPt, &I);
    }
  } else if (ExitBlocks.size() == 0) {
    // The instruction is actually dead if there ARE NO exit blocks.
    I.getParent()->getInstList().erase(&I);
    return;   // Don't count this as a sunk instruction, don't check operands.
  } else {
    // Otherwise, if we have multiple exits, use the PromoteMem2Reg function to
    // do all of the hard work of inserting PHI nodes as necessary.  We convert
    // the value into a stack object to get it to do this.

    // Firstly, we create a stack object to hold the value...
    AllocaInst *AI = new AllocaInst(I.getType(), 0, I.getName(),
                                   I.getParent()->getParent()->front().begin());

    // Secondly, insert load instructions for each use of the instruction
    // outside of the loop.
    while (!I.use_empty()) {
      Instruction *U = cast<Instruction>(I.use_back());

      // If the user is a PHI Node, we actually have to insert load instructions
      // in all predecessor blocks, not in the PHI block itself!
      if (PHINode *UPN = dyn_cast<PHINode>(U)) {
        // Only insert into each predecessor once, so that we don't have
        // different incoming values from the same block!
        std::map<BasicBlock*, Value*> InsertedBlocks;
        for (unsigned i = 0, e = UPN->getNumIncomingValues(); i != e; ++i)
          if (UPN->getIncomingValue(i) == &I) {
            BasicBlock *Pred = UPN->getIncomingBlock(i);
            Value *&PredVal = InsertedBlocks[Pred];
            if (!PredVal) {
              // Insert a new load instruction right before the terminator in
              // the predecessor block.
              PredVal = new LoadInst(AI, "", Pred->getTerminator());
            }

            UPN->setIncomingValue(i, PredVal);
          }

      } else {
        LoadInst *L = new LoadInst(AI, "", U);
        U->replaceUsesOfWith(&I, L);
      }
    }

    // Thirdly, insert a copy of the instruction in each exit block of the loop
    // that is dominated by the instruction, storing the result into the memory
    // location.  Be careful not to insert the instruction into any particular
    // basic block more than once.
    std::set<BasicBlock*> InsertedBlocks;
    BasicBlock *InstOrigBB = I.getParent();

    for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i) {
      BasicBlock *ExitBlock = ExitBlocks[i];

      if (isExitBlockDominatedByBlockInLoop(ExitBlock, InstOrigBB)) {
        std::set<BasicBlock*>::iterator SI =
          InsertedBlocks.lower_bound(ExitBlock);
        // If we haven't already processed this exit block, do so now.
        if (SI == InsertedBlocks.end() || *SI != ExitBlock) {
          // Insert the code after the last PHI node...
          BasicBlock::iterator InsertPt = ExitBlock->begin();
          while (isa<PHINode>(InsertPt)) ++InsertPt;
          
          // If this is the first exit block processed, just move the original
          // instruction, otherwise clone the original instruction and insert
          // the copy.
          Instruction *New;
          if (InsertedBlocks.empty()) {
            I.getParent()->getInstList().remove(&I);
            ExitBlock->getInstList().insert(InsertPt, &I);
            New = &I;
          } else {
            New = I.clone();
            New->setName(I.getName()+".le");
            ExitBlock->getInstList().insert(InsertPt, New);
          }
          
          // Now that we have inserted the instruction, store it into the alloca
          new StoreInst(New, AI, InsertPt);
          
          // Remember we processed this block
          InsertedBlocks.insert(SI, ExitBlock);
        }
      }
    }
      
    // Finally, promote the fine value to SSA form.
    std::vector<AllocaInst*> Allocas;
    Allocas.push_back(AI);
    PromoteMemToReg(Allocas, *DT, *DF, AA->getTargetData());
  }
  
  if (isa<LoadInst>(I)) ++NumMovedLoads;
  ++NumSunk;
  Changed = true;

  // Since we just sunk an instruction, check to see if any other instructions
  // used by this instruction are now sinkable.  If so, sink them too.
  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i)
    if (Instruction *OpI = dyn_cast<Instruction>(I.getOperand(i)))
      if (CurLoop->contains(OpI->getParent()) && canSinkOrHoistInst(*OpI) &&
          isNotUsedInLoop(*OpI) &&
          isSafeToExecuteUnconditionally(*OpI))
        sink(*OpI);
}

/// hoist - When an instruction is found to only use loop invariant operands
/// that is safe to hoist, this instruction is called to do the dirty work.
///
void LICM::hoist(Instruction &I) {
  DEBUG(std::cerr << "LICM hoisting to";
        WriteAsOperand(std::cerr, Preheader, false);
        std::cerr << ": " << I);

  // Remove the instruction from its current basic block... but don't delete the
  // instruction.
  I.getParent()->getInstList().remove(&I);

  // Insert the new node in Preheader, before the terminator.
  Preheader->getInstList().insert(Preheader->getTerminator(), &I);
  
  if (isa<LoadInst>(I)) ++NumMovedLoads;
  ++NumHoisted;
  Changed = true;
}

/// isSafeToExecuteUnconditionally - Only sink or hoist an instruction if it is
/// not a trapping instruction or if it is a trapping instruction and is
/// guaranteed to execute.
///
bool LICM::isSafeToExecuteUnconditionally(Instruction &Inst) {
  // If it is not a trapping instruction, it is always safe to hoist.
  if (!Inst.isTrapping()) return true;
  
  // Otherwise we have to check to make sure that the instruction dominates all
  // of the exit blocks.  If it doesn't, then there is a path out of the loop
  // which does not execute this instruction, so we can't hoist it.

  // If the instruction is in the header block for the loop (which is very
  // common), it is always guaranteed to dominate the exit blocks.  Since this
  // is a common case, and can save some work, check it now.
  if (Inst.getParent() == CurLoop->getHeader())
    return true;

  // Get the exit blocks for the current loop.
  const std::vector<BasicBlock*> &ExitBlocks = CurLoop->getExitBlocks();

  // For each exit block, get the DT node and walk up the DT until the
  // instruction's basic block is found or we exit the loop.
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
    if (!isExitBlockDominatedByBlockInLoop(ExitBlocks[i], Inst.getParent()))
      return false;
  
  return true;
}


/// PromoteValuesInLoop - Try to promote memory values to scalars by sinking
/// stores out of the loop and moving loads to before the loop.  We do this by
/// looping over the stores in the loop, looking for stores to Must pointers
/// which are loop invariant.  We promote these memory locations to use allocas
/// instead.  These allocas can easily be raised to register values by the
/// PromoteMem2Reg functionality.
///
void LICM::PromoteValuesInLoop() {
  // PromotedValues - List of values that are promoted out of the loop.  Each
  // value has an alloca instruction for it, and a canonical version of the
  // pointer.
  std::vector<std::pair<AllocaInst*, Value*> > PromotedValues;
  std::map<Value*, AllocaInst*> ValueToAllocaMap; // Map of ptr to alloca

  findPromotableValuesInLoop(PromotedValues, ValueToAllocaMap);
  if (ValueToAllocaMap.empty()) return;   // If there are values to promote...

  Changed = true;
  NumPromoted += PromotedValues.size();

  // Emit a copy from the value into the alloca'd value in the loop preheader
  TerminatorInst *LoopPredInst = Preheader->getTerminator();
  for (unsigned i = 0, e = PromotedValues.size(); i != e; ++i) {
    // Load from the memory we are promoting...
    LoadInst *LI = new LoadInst(PromotedValues[i].second, 
                                PromotedValues[i].second->getName()+".promoted",
                                LoopPredInst);
    // Store into the temporary alloca...
    new StoreInst(LI, PromotedValues[i].first, LoopPredInst);
  }
  
  // Scan the basic blocks in the loop, replacing uses of our pointers with
  // uses of the allocas in question.  If we find a branch that exits the
  // loop, make sure to put reload code into all of the successors of the
  // loop.
  //
  const std::vector<BasicBlock*> &LoopBBs = CurLoop->getBlocks();
  for (std::vector<BasicBlock*>::const_iterator I = LoopBBs.begin(),
         E = LoopBBs.end(); I != E; ++I) {
    // Rewrite all loads and stores in the block of the pointer...
    for (BasicBlock::iterator II = (*I)->begin(), E = (*I)->end();
         II != E; ++II) {
      if (LoadInst *L = dyn_cast<LoadInst>(II)) {
        std::map<Value*, AllocaInst*>::iterator
          I = ValueToAllocaMap.find(L->getOperand(0));
        if (I != ValueToAllocaMap.end())
          L->setOperand(0, I->second);    // Rewrite load instruction...
      } else if (StoreInst *S = dyn_cast<StoreInst>(II)) {
        std::map<Value*, AllocaInst*>::iterator
          I = ValueToAllocaMap.find(S->getOperand(1));
        if (I != ValueToAllocaMap.end())
          S->setOperand(1, I->second);    // Rewrite store instruction...
      }
    }

    // Check to see if any successors of this block are outside of the loop.
    // If so, we need to copy the value from the alloca back into the memory
    // location...
    //
    for (succ_iterator SI = succ_begin(*I), SE = succ_end(*I); SI != SE; ++SI)
      if (!CurLoop->contains(*SI)) {
        // Copy all of the allocas into their memory locations...
        BasicBlock::iterator BI = (*SI)->begin();
        while (isa<PHINode>(*BI))
          ++BI;             // Skip over all of the phi nodes in the block...
        Instruction *InsertPos = BI;
        for (unsigned i = 0, e = PromotedValues.size(); i != e; ++i) {
          // Load from the alloca...
          LoadInst *LI = new LoadInst(PromotedValues[i].first, "", InsertPos);
          // Store into the memory we promoted...
          new StoreInst(LI, PromotedValues[i].second, InsertPos);
        }
      }
  }

  // Now that we have done the deed, use the mem2reg functionality to promote
  // all of the new allocas we just created into real SSA registers...
  //
  std::vector<AllocaInst*> PromotedAllocas;
  PromotedAllocas.reserve(PromotedValues.size());
  for (unsigned i = 0, e = PromotedValues.size(); i != e; ++i)
    PromotedAllocas.push_back(PromotedValues[i].first);
  PromoteMemToReg(PromotedAllocas, *DT, *DF, AA->getTargetData());
}

/// findPromotableValuesInLoop - Check the current loop for stores to definite
/// pointers, which are not loaded and stored through may aliases.  If these are
/// found, create an alloca for the value, add it to the PromotedValues list,
/// and keep track of the mapping from value to alloca...
///
void LICM::findPromotableValuesInLoop(
                   std::vector<std::pair<AllocaInst*, Value*> > &PromotedValues,
                             std::map<Value*, AllocaInst*> &ValueToAllocaMap) {
  Instruction *FnStart = CurLoop->getHeader()->getParent()->begin()->begin();

  // Loop over all of the alias sets in the tracker object...
  for (AliasSetTracker::iterator I = CurAST->begin(), E = CurAST->end();
       I != E; ++I) {
    AliasSet &AS = *I;
    // We can promote this alias set if it has a store, if it is a "Must" alias
    // set, and if the pointer is loop invariant.
    if (!AS.isForwardingAliasSet() && AS.isMod() && AS.isMustAlias() &&
        isLoopInvariant(AS.begin()->first)) {
      assert(AS.begin() != AS.end() &&
             "Must alias set should have at least one pointer element in it!");
      Value *V = AS.begin()->first;

      // Check that all of the pointers in the alias set have the same type.  We
      // cannot (yet) promote a memory location that is loaded and stored in
      // different sizes.
      bool PointerOk = true;
      for (AliasSet::iterator I = AS.begin(), E = AS.end(); I != E; ++I)
        if (V->getType() != I->first->getType()) {
          PointerOk = false;
          break;
        }

      if (PointerOk) {
        const Type *Ty = cast<PointerType>(V->getType())->getElementType();
        AllocaInst *AI = new AllocaInst(Ty, 0, V->getName()+".tmp", FnStart);
        PromotedValues.push_back(std::make_pair(AI, V));
        
        for (AliasSet::iterator I = AS.begin(), E = AS.end(); I != E; ++I)
          ValueToAllocaMap.insert(std::make_pair(I->first, AI));
        
        DEBUG(std::cerr << "LICM: Promoting value: " << *V << "\n");
      }
    }
  }
}
