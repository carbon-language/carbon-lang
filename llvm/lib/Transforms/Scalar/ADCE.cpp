//===- ADCE.cpp - Code to perform aggressive dead code elimination --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements "aggressive" dead code elimination.  ADCE is DCe where
// values are assumed to be dead until proven otherwise.  This is similar to
// SCCP, except applied to the liveness of values.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Support/CFG.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <iostream>
using namespace llvm;

namespace {
  Statistic<> NumBlockRemoved("adce", "Number of basic blocks removed");
  Statistic<> NumInstRemoved ("adce", "Number of instructions removed");
  Statistic<> NumCallRemoved ("adce", "Number of calls and invokes removed");

//===----------------------------------------------------------------------===//
// ADCE Class
//
// This class does all of the work of Aggressive Dead Code Elimination.
// It's public interface consists of a constructor and a doADCE() method.
//
class ADCE : public FunctionPass {
  Function *Func;                       // The function that we are working on
  std::vector<Instruction*> WorkList;   // Instructions that just became live
  std::set<Instruction*>    LiveSet;    // The set of live instructions

  //===--------------------------------------------------------------------===//
  // The public interface for this class
  //
public:
  // Execute the Aggressive Dead Code Elimination Algorithm
  //
  virtual bool runOnFunction(Function &F) {
    Func = &F;
    bool Changed = doADCE();
    assert(WorkList.empty());
    LiveSet.clear();
    return Changed;
  }
  // getAnalysisUsage - We require post dominance frontiers (aka Control
  // Dependence Graph)
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    // We require that all function nodes are unified, because otherwise code
    // can be marked live that wouldn't necessarily be otherwise.
    AU.addRequired<UnifyFunctionExitNodes>();
    AU.addRequired<AliasAnalysis>();
    AU.addRequired<PostDominatorTree>();
    AU.addRequired<PostDominanceFrontier>();
  }


  //===--------------------------------------------------------------------===//
  // The implementation of this class
  //
private:
  // doADCE() - Run the Aggressive Dead Code Elimination algorithm, returning
  // true if the function was modified.
  //
  bool doADCE();

  void markBlockAlive(BasicBlock *BB);


  // deleteDeadInstructionsInLiveBlock - Loop over all of the instructions in
  // the specified basic block, deleting ones that are dead according to
  // LiveSet.
  bool deleteDeadInstructionsInLiveBlock(BasicBlock *BB);

  TerminatorInst *convertToUnconditionalBranch(TerminatorInst *TI);

  inline void markInstructionLive(Instruction *I) {
    if (!LiveSet.insert(I).second) return;
    DEBUG(std::cerr << "Insn Live: " << *I);
    WorkList.push_back(I);
  }

  inline void markTerminatorLive(const BasicBlock *BB) {
    DEBUG(std::cerr << "Terminator Live: " << *BB->getTerminator());
    markInstructionLive(const_cast<TerminatorInst*>(BB->getTerminator()));
  }
};

  RegisterPass<ADCE> X("adce", "Aggressive Dead Code Elimination");
} // End of anonymous namespace

FunctionPass *llvm::createAggressiveDCEPass() { return new ADCE(); }

void ADCE::markBlockAlive(BasicBlock *BB) {
  // Mark the basic block as being newly ALIVE... and mark all branches that
  // this block is control dependent on as being alive also...
  //
  PostDominanceFrontier &CDG = getAnalysis<PostDominanceFrontier>();

  PostDominanceFrontier::const_iterator It = CDG.find(BB);
  if (It != CDG.end()) {
    // Get the blocks that this node is control dependent on...
    const PostDominanceFrontier::DomSetType &CDB = It->second;
    for (PostDominanceFrontier::DomSetType::const_iterator I =
           CDB.begin(), E = CDB.end(); I != E; ++I)
      markTerminatorLive(*I);   // Mark all their terminators as live
  }

  // If this basic block is live, and it ends in an unconditional branch, then
  // the branch is alive as well...
  if (BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator()))
    if (BI->isUnconditional())
      markTerminatorLive(BB);
}

// deleteDeadInstructionsInLiveBlock - Loop over all of the instructions in the
// specified basic block, deleting ones that are dead according to LiveSet.
bool ADCE::deleteDeadInstructionsInLiveBlock(BasicBlock *BB) {
  bool Changed = false;
  for (BasicBlock::iterator II = BB->begin(), E = --BB->end(); II != E; ) {
    Instruction *I = II++;
    if (!LiveSet.count(I)) {              // Is this instruction alive?
      if (!I->use_empty())
        I->replaceAllUsesWith(UndefValue::get(I->getType()));

      // Nope... remove the instruction from it's basic block...
      if (isa<CallInst>(I))
        ++NumCallRemoved;
      else
        ++NumInstRemoved;
      BB->getInstList().erase(I);
      Changed = true;
    }
  }
  return Changed;
}


/// convertToUnconditionalBranch - Transform this conditional terminator
/// instruction into an unconditional branch because we don't care which of the
/// successors it goes to.  This eliminate a use of the condition as well.
///
TerminatorInst *ADCE::convertToUnconditionalBranch(TerminatorInst *TI) {
  BranchInst *NB = new BranchInst(TI->getSuccessor(0), TI);
  BasicBlock *BB = TI->getParent();

  // Remove entries from PHI nodes to avoid confusing ourself later...
  for (unsigned i = 1, e = TI->getNumSuccessors(); i != e; ++i)
    TI->getSuccessor(i)->removePredecessor(BB);

  // Delete the old branch itself...
  BB->getInstList().erase(TI);
  return NB;
}


// doADCE() - Run the Aggressive Dead Code Elimination algorithm, returning
// true if the function was modified.
//
bool ADCE::doADCE() {
  bool MadeChanges = false;

  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();


  // Iterate over all invokes in the function, turning invokes into calls if
  // they cannot throw.
  for (Function::iterator BB = Func->begin(), E = Func->end(); BB != E; ++BB)
    if (InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator()))
      if (Function *F = II->getCalledFunction())
        if (AA.onlyReadsMemory(F)) {
          // The function cannot unwind.  Convert it to a call with a branch
          // after it to the normal destination.
          std::vector<Value*> Args(II->op_begin()+3, II->op_end());
          std::string Name = II->getName(); II->setName("");
          CallInst *NewCall = new CallInst(F, Args, Name, II);
          NewCall->setCallingConv(II->getCallingConv());
          II->replaceAllUsesWith(NewCall);
          new BranchInst(II->getNormalDest(), II);

          // Update PHI nodes in the unwind destination
          II->getUnwindDest()->removePredecessor(BB);
          BB->getInstList().erase(II);

          if (NewCall->use_empty()) {
            BB->getInstList().erase(NewCall);
            ++NumCallRemoved;
          }
        }

  // Iterate over all of the instructions in the function, eliminating trivially
  // dead instructions, and marking instructions live that are known to be
  // needed.  Perform the walk in depth first order so that we avoid marking any
  // instructions live in basic blocks that are unreachable.  These blocks will
  // be eliminated later, along with the instructions inside.
  //
  std::set<BasicBlock*> ReachableBBs;
  for (df_ext_iterator<BasicBlock*>
         BBI = df_ext_begin(&Func->front(), ReachableBBs),
         BBE = df_ext_end(&Func->front(), ReachableBBs); BBI != BBE; ++BBI) {
    BasicBlock *BB = *BBI;
    for (BasicBlock::iterator II = BB->begin(), EI = BB->end(); II != EI; ) {
      Instruction *I = II++;
      if (CallInst *CI = dyn_cast<CallInst>(I)) {
        Function *F = CI->getCalledFunction();
        if (F && AA.onlyReadsMemory(F)) {
          if (CI->use_empty()) {
            BB->getInstList().erase(CI);
            ++NumCallRemoved;
          }
        } else {
          markInstructionLive(I);
        }
      } else if (I->mayWriteToMemory() || isa<ReturnInst>(I) ||
                 isa<UnwindInst>(I) || isa<UnreachableInst>(I)) {
        // FIXME: Unreachable instructions should not be marked intrinsically
        // live here.
        markInstructionLive(I);
      } else if (isInstructionTriviallyDead(I)) {
        // Remove the instruction from it's basic block...
        BB->getInstList().erase(I);
        ++NumInstRemoved;
      }
    }
  }

  // Check to ensure we have an exit node for this CFG.  If we don't, we won't
  // have any post-dominance information, thus we cannot perform our
  // transformations safely.
  //
  PostDominatorTree &DT = getAnalysis<PostDominatorTree>();
  if (DT[&Func->getEntryBlock()] == 0) {
    WorkList.clear();
    return MadeChanges;
  }

  // Scan the function marking blocks without post-dominance information as
  // live.  Blocks without post-dominance information occur when there is an
  // infinite loop in the program.  Because the infinite loop could contain a
  // function which unwinds, exits or has side-effects, we don't want to delete
  // the infinite loop or those blocks leading up to it.
  for (Function::iterator I = Func->begin(), E = Func->end(); I != E; ++I)
    if (DT[I] == 0 && ReachableBBs.count(I))
      for (pred_iterator PI = pred_begin(I), E = pred_end(I); PI != E; ++PI)
        markInstructionLive((*PI)->getTerminator());

  DEBUG(std::cerr << "Processing work list\n");

  // AliveBlocks - Set of basic blocks that we know have instructions that are
  // alive in them...
  //
  std::set<BasicBlock*> AliveBlocks;

  // Process the work list of instructions that just became live... if they
  // became live, then that means that all of their operands are necessary as
  // well... make them live as well.
  //
  while (!WorkList.empty()) {
    Instruction *I = WorkList.back(); // Get an instruction that became live...
    WorkList.pop_back();

    BasicBlock *BB = I->getParent();
    if (!ReachableBBs.count(BB)) continue;
    if (AliveBlocks.insert(BB).second)     // Basic block not alive yet.
      markBlockAlive(BB);             // Make it so now!

    // PHI nodes are a special case, because the incoming values are actually
    // defined in the predecessor nodes of this block, meaning that the PHI
    // makes the predecessors alive.
    //
    if (PHINode *PN = dyn_cast<PHINode>(I)) {
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        // If the incoming edge is clearly dead, it won't have control
        // dependence information.  Do not mark it live.
        BasicBlock *PredBB = PN->getIncomingBlock(i);
        if (ReachableBBs.count(PredBB)) {
          // FIXME: This should mark the control dependent edge as live, not
          // necessarily the predecessor itself!
          if (AliveBlocks.insert(PredBB).second)
            markBlockAlive(PN->getIncomingBlock(i));   // Block is newly ALIVE!
          if (Instruction *Op = dyn_cast<Instruction>(PN->getIncomingValue(i)))
            markInstructionLive(Op);
        }
      }
    } else {
      // Loop over all of the operands of the live instruction, making sure that
      // they are known to be alive as well.
      //
      for (unsigned op = 0, End = I->getNumOperands(); op != End; ++op)
        if (Instruction *Operand = dyn_cast<Instruction>(I->getOperand(op)))
          markInstructionLive(Operand);
    }
  }

  DEBUG(
    std::cerr << "Current Function: X = Live\n";
    for (Function::iterator I = Func->begin(), E = Func->end(); I != E; ++I){
      std::cerr << I->getName() << ":\t"
                << (AliveBlocks.count(I) ? "LIVE\n" : "DEAD\n");
      for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI){
        if (LiveSet.count(BI)) std::cerr << "X ";
        std::cerr << *BI;
      }
    });

  // All blocks being live is a common case, handle it specially.
  if (AliveBlocks.size() == Func->size()) {  // No dead blocks?
    for (Function::iterator I = Func->begin(), E = Func->end(); I != E; ++I) {
      // Loop over all of the instructions in the function deleting instructions
      // to drop their references.
      deleteDeadInstructionsInLiveBlock(I);

      // Check to make sure the terminator instruction is live.  If it isn't,
      // this means that the condition that it branches on (we know it is not an
      // unconditional branch), is not needed to make the decision of where to
      // go to, because all outgoing edges go to the same place.  We must remove
      // the use of the condition (because it's probably dead), so we convert
      // the terminator to an unconditional branch.
      //
      TerminatorInst *TI = I->getTerminator();
      if (!LiveSet.count(TI))
        convertToUnconditionalBranch(TI);
    }

    return MadeChanges;
  }


  // If the entry node is dead, insert a new entry node to eliminate the entry
  // node as a special case.
  //
  if (!AliveBlocks.count(&Func->front())) {
    BasicBlock *NewEntry = new BasicBlock();
    new BranchInst(&Func->front(), NewEntry);
    Func->getBasicBlockList().push_front(NewEntry);
    AliveBlocks.insert(NewEntry);    // This block is always alive!
    LiveSet.insert(NewEntry->getTerminator());  // The branch is live
  }

  // Loop over all of the alive blocks in the function.  If any successor
  // blocks are not alive, we adjust the outgoing branches to branch to the
  // first live postdominator of the live block, adjusting any PHI nodes in
  // the block to reflect this.
  //
  for (Function::iterator I = Func->begin(), E = Func->end(); I != E; ++I)
    if (AliveBlocks.count(I)) {
      BasicBlock *BB = I;
      TerminatorInst *TI = BB->getTerminator();

      // If the terminator instruction is alive, but the block it is contained
      // in IS alive, this means that this terminator is a conditional branch on
      // a condition that doesn't matter.  Make it an unconditional branch to
      // ONE of the successors.  This has the side effect of dropping a use of
      // the conditional value, which may also be dead.
      if (!LiveSet.count(TI))
        TI = convertToUnconditionalBranch(TI);

      // Loop over all of the successors, looking for ones that are not alive.
      // We cannot save the number of successors in the terminator instruction
      // here because we may remove them if we don't have a postdominator.
      //
      for (unsigned i = 0; i != TI->getNumSuccessors(); ++i)
        if (!AliveBlocks.count(TI->getSuccessor(i))) {
          // Scan up the postdominator tree, looking for the first
          // postdominator that is alive, and the last postdominator that is
          // dead...
          //
          PostDominatorTree::Node *LastNode = DT[TI->getSuccessor(i)];
          PostDominatorTree::Node *NextNode = 0;

          if (LastNode) {
            NextNode = LastNode->getIDom();
            while (!AliveBlocks.count(NextNode->getBlock())) {
              LastNode = NextNode;
              NextNode = NextNode->getIDom();
              if (NextNode == 0) {
                LastNode = 0;
                break;
              }
            }
          }

          // There is a special case here... if there IS no post-dominator for
          // the block we have nowhere to point our branch to.  Instead, convert
          // it to a return.  This can only happen if the code branched into an
          // infinite loop.  Note that this may not be desirable, because we
          // _are_ altering the behavior of the code.  This is a well known
          // drawback of ADCE, so in the future if we choose to revisit the
          // decision, this is where it should be.
          //
          if (LastNode == 0) {        // No postdominator!
            if (!isa<InvokeInst>(TI)) {
              // Call RemoveSuccessor to transmogrify the terminator instruction
              // to not contain the outgoing branch, or to create a new
              // terminator if the form fundamentally changes (i.e.,
              // unconditional branch to return).  Note that this will change a
              // branch into an infinite loop into a return instruction!
              //
              RemoveSuccessor(TI, i);

              // RemoveSuccessor may replace TI... make sure we have a fresh
              // pointer.
              //
              TI = BB->getTerminator();

              // Rescan this successor...
              --i;
            } else {

            }
          } else {
            // Get the basic blocks that we need...
            BasicBlock *LastDead = LastNode->getBlock();
            BasicBlock *NextAlive = NextNode->getBlock();

            // Make the conditional branch now go to the next alive block...
            TI->getSuccessor(i)->removePredecessor(BB);
            TI->setSuccessor(i, NextAlive);

            // If there are PHI nodes in NextAlive, we need to add entries to
            // the PHI nodes for the new incoming edge.  The incoming values
            // should be identical to the incoming values for LastDead.
            //
            for (BasicBlock::iterator II = NextAlive->begin();
                 isa<PHINode>(II); ++II) {
              PHINode *PN = cast<PHINode>(II);
              if (LiveSet.count(PN)) {  // Only modify live phi nodes
                // Get the incoming value for LastDead...
                int OldIdx = PN->getBasicBlockIndex(LastDead);
                assert(OldIdx != -1 &&"LastDead is not a pred of NextAlive!");
                Value *InVal = PN->getIncomingValue(OldIdx);

                // Add an incoming value for BB now...
                PN->addIncoming(InVal, BB);
              }
            }
          }
        }

      // Now loop over all of the instructions in the basic block, deleting
      // dead instructions.  This is so that the next sweep over the program
      // can safely delete dead instructions without other dead instructions
      // still referring to them.
      //
      deleteDeadInstructionsInLiveBlock(BB);
    }

  // Loop over all of the basic blocks in the function, dropping references of
  // the dead basic blocks.  We must do this after the previous step to avoid
  // dropping references to PHIs which still have entries...
  //
  std::vector<BasicBlock*> DeadBlocks;
  for (Function::iterator BB = Func->begin(), E = Func->end(); BB != E; ++BB)
    if (!AliveBlocks.count(BB)) {
      // Remove PHI node entries for this block in live successor blocks.
      for (succ_iterator SI = succ_begin(BB), E = succ_end(BB); SI != E; ++SI)
        if (!SI->empty() && isa<PHINode>(SI->front()) && AliveBlocks.count(*SI))
          (*SI)->removePredecessor(BB);

      BB->dropAllReferences();
      MadeChanges = true;
      DeadBlocks.push_back(BB);
    }

  NumBlockRemoved += DeadBlocks.size();

  // Now loop through all of the blocks and delete the dead ones.  We can safely
  // do this now because we know that there are no references to dead blocks
  // (because they have dropped all of their references).
  for (std::vector<BasicBlock*>::iterator I = DeadBlocks.begin(),
         E = DeadBlocks.end(); I != E; ++I)
    Func->getBasicBlockList().erase(*I);

  return MadeChanges;
}
