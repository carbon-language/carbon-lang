//===- ADCE.cpp - Code to perform aggressive dead code elimination --------===//
//
// This file implements "aggressive" dead code elimination.  ADCE is DCe where
// values are assumed to be dead until proven otherwise.  This is similar to 
// SCCP, except applied to the liveness of values.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Type.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/Constant.h"
#include "llvm/Support/CFG.h"
#include "Support/STLExtras.h"
#include "Support/DepthFirstIterator.h"
#include "Support/Statistic.h"
#include <algorithm>

namespace {
  Statistic<> NumBlockRemoved("adce", "Number of basic blocks removed");
  Statistic<> NumInstRemoved ("adce", "Number of instructions removed");

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


  // dropReferencesOfDeadInstructionsInLiveBlock - Loop over all of the
  // instructions in the specified basic block, dropping references on
  // instructions that are dead according to LiveSet.
  bool dropReferencesOfDeadInstructionsInLiveBlock(BasicBlock *BB);

  TerminatorInst *convertToUnconditionalBranch(TerminatorInst *TI);

  inline void markInstructionLive(Instruction *I) {
    if (LiveSet.count(I)) return;
    DEBUG(std::cerr << "Insn Live: " << I);
    LiveSet.insert(I);
    WorkList.push_back(I);
  }

  inline void markTerminatorLive(const BasicBlock *BB) {
    DEBUG(std::cerr << "Terminat Live: " << BB->getTerminator());
    markInstructionLive((Instruction*)BB->getTerminator());
  }
};

  RegisterOpt<ADCE> X("adce", "Aggressive Dead Code Elimination");
} // End of anonymous namespace

Pass *createAggressiveDCEPass() { return new ADCE(); }

void ADCE::markBlockAlive(BasicBlock *BB) {
  // Mark the basic block as being newly ALIVE... and mark all branches that
  // this block is control dependant on as being alive also...
  //
  PostDominanceFrontier &CDG = getAnalysis<PostDominanceFrontier>();

  PostDominanceFrontier::const_iterator It = CDG.find(BB);
  if (It != CDG.end()) {
    // Get the blocks that this node is control dependant on...
    const PostDominanceFrontier::DomSetType &CDB = It->second;
    for_each(CDB.begin(), CDB.end(),   // Mark all their terminators as live
             bind_obj(this, &ADCE::markTerminatorLive));
  }
  
  // If this basic block is live, and it ends in an unconditional branch, then
  // the branch is alive as well...
  if (BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator()))
    if (BI->isUnconditional())
      markTerminatorLive(BB);
}

// dropReferencesOfDeadInstructionsInLiveBlock - Loop over all of the
// instructions in the specified basic block, dropping references on
// instructions that are dead according to LiveSet.
bool ADCE::dropReferencesOfDeadInstructionsInLiveBlock(BasicBlock *BB) {
  bool Changed = false;
  for (BasicBlock::iterator I = BB->begin(), E = --BB->end(); I != E; )
    if (!LiveSet.count(I)) {              // Is this instruction alive?
      I->dropAllReferences();             // Nope, drop references... 
      if (PHINode *PN = dyn_cast<PHINode>(I)) {
        // We don't want to leave PHI nodes in the program that have
        // #arguments != #predecessors, so we remove them now.
        //
        PN->replaceAllUsesWith(Constant::getNullValue(PN->getType()));
        
        // Delete the instruction...
        I = BB->getInstList().erase(I);
        Changed = true;
      } else {
        ++I;
      }
    } else {
      ++I;
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

  // Iterate over all of the instructions in the function, eliminating trivially
  // dead instructions, and marking instructions live that are known to be 
  // needed.  Perform the walk in depth first order so that we avoid marking any
  // instructions live in basic blocks that are unreachable.  These blocks will
  // be eliminated later, along with the instructions inside.
  //
  for (df_iterator<Function*> BBI = df_begin(Func), BBE = df_end(Func);
       BBI != BBE; ++BBI) {
    BasicBlock *BB = *BBI;
    for (BasicBlock::iterator II = BB->begin(), EI = BB->end(); II != EI; ) {
      if (II->mayWriteToMemory() || II->getOpcode() == Instruction::Ret) {
	markInstructionLive(II);
        ++II;  // Increment the inst iterator if the inst wasn't deleted
      } else if (isInstructionTriviallyDead(II)) {
        // Remove the instruction from it's basic block...
        II = BB->getInstList().erase(II);
        ++NumInstRemoved;
        MadeChanges = true;
      } else {
        ++II;  // Increment the inst iterator if the inst wasn't deleted
      }
    }
  }

  // Check to ensure we have an exit node for this CFG.  If we don't, we won't
  // have any post-dominance information, thus we cannot perform our
  // transformations safely.
  //
  PostDominatorTree &DT = getAnalysis<PostDominatorTree>();
  if (DT[&Func->getEntryNode()] == 0) {
    WorkList.clear();
    return MadeChanges;
  }

  DEBUG(std::cerr << "Processing work list\n");

  // AliveBlocks - Set of basic blocks that we know have instructions that are
  // alive in them...
  //
  std::set<BasicBlock*> AliveBlocks;

  // Process the work list of instructions that just became live... if they
  // became live, then that means that all of their operands are neccesary as
  // well... make them live as well.
  //
  while (!WorkList.empty()) {
    Instruction *I = WorkList.back(); // Get an instruction that became live...
    WorkList.pop_back();

    BasicBlock *BB = I->getParent();
    if (!AliveBlocks.count(BB)) {     // Basic block not alive yet...
      AliveBlocks.insert(BB);         // Block is now ALIVE!
      markBlockAlive(BB);             // Make it so now!
    }

    // PHI nodes are a special case, because the incoming values are actually
    // defined in the predecessor nodes of this block, meaning that the PHI
    // makes the predecessors alive.
    //
    if (PHINode *PN = dyn_cast<PHINode>(I))
      for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI)
        if (!AliveBlocks.count(*PI)) {
          AliveBlocks.insert(BB);         // Block is now ALIVE!
          markBlockAlive(*PI);
        }

    // Loop over all of the operands of the live instruction, making sure that
    // they are known to be alive as well...
    //
    for (unsigned op = 0, End = I->getNumOperands(); op != End; ++op)
      if (Instruction *Operand = dyn_cast<Instruction>(I->getOperand(op)))
	markInstructionLive(Operand);
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

  // Find the first postdominator of the entry node that is alive.  Make it the
  // new entry node...
  //
  if (AliveBlocks.size() == Func->size()) {  // No dead blocks?
    for (Function::iterator I = Func->begin(), E = Func->end(); I != E; ++I) {
      // Loop over all of the instructions in the function, telling dead
      // instructions to drop their references.  This is so that the next sweep
      // over the program can safely delete dead instructions without other dead
      // instructions still refering to them.
      //
      dropReferencesOfDeadInstructionsInLiveBlock(I);

      // Check to make sure the terminator instruction is live.  If it isn't,
      // this means that the condition that it branches on (we know it is not an
      // unconditional branch), is not needed to make the decision of where to
      // go to, because all outgoing edges go to the same place.  We must remove
      // the use of the condition (because it's probably dead), so we convert
      // the terminator to a conditional branch.
      //
      TerminatorInst *TI = I->getTerminator();
      if (!LiveSet.count(TI))
        convertToUnconditionalBranch(TI);
    }
    
  } else {                                   // If there are some blocks dead...
    // If the entry node is dead, insert a new entry node to eliminate the entry
    // node as a special case.
    //
    if (!AliveBlocks.count(&Func->front())) {
      BasicBlock *NewEntry = new BasicBlock();
      NewEntry->getInstList().push_back(new BranchInst(&Func->front()));
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
        // in IS alive, this means that this terminator is a conditional branch
        // on a condition that doesn't matter.  Make it an unconditional branch
        // to ONE of the successors.  This has the side effect of dropping a use
        // of the conditional value, which may also be dead.
        if (!LiveSet.count(TI))
          TI = convertToUnconditionalBranch(TI);

        // Loop over all of the successors, looking for ones that are not alive.
        // We cannot save the number of successors in the terminator instruction
        // here because we may remove them if we don't have a postdominator...
        //
        for (unsigned i = 0; i != TI->getNumSuccessors(); ++i)
          if (!AliveBlocks.count(TI->getSuccessor(i))) {
            // Scan up the postdominator tree, looking for the first
            // postdominator that is alive, and the last postdominator that is
            // dead...
            //
            PostDominatorTree::Node *LastNode = DT[TI->getSuccessor(i)];

            // There is a special case here... if there IS no post-dominator for
            // the block we have no owhere to point our branch to.  Instead,
            // convert it to a return.  This can only happen if the code
            // branched into an infinite loop.  Note that this may not be
            // desirable, because we _are_ altering the behavior of the code.
            // This is a well known drawback of ADCE, so in the future if we
            // choose to revisit the decision, this is where it should be.
            //
            if (LastNode == 0) {        // No postdominator!
              // Call RemoveSuccessor to transmogrify the terminator instruction
              // to not contain the outgoing branch, or to create a new
              // terminator if the form fundementally changes (ie unconditional
              // branch to return).  Note that this will change a branch into an
              // infinite loop into a return instruction!
              //
              RemoveSuccessor(TI, i);

              // RemoveSuccessor may replace TI... make sure we have a fresh
              // pointer... and e variable.
              //
              TI = BB->getTerminator();

              // Rescan this successor...
              --i;
            } else {
              PostDominatorTree::Node *NextNode = LastNode->getIDom();

              while (!AliveBlocks.count(NextNode->getNode())) {
                LastNode = NextNode;
                NextNode = NextNode->getIDom();
              }
            
              // Get the basic blocks that we need...
              BasicBlock *LastDead = LastNode->getNode();
              BasicBlock *NextAlive = NextNode->getNode();

              // Make the conditional branch now go to the next alive block...
              TI->getSuccessor(i)->removePredecessor(BB);
              TI->setSuccessor(i, NextAlive);

              // If there are PHI nodes in NextAlive, we need to add entries to
              // the PHI nodes for the new incoming edge.  The incoming values
              // should be identical to the incoming values for LastDead.
              //
              for (BasicBlock::iterator II = NextAlive->begin();
                   PHINode *PN = dyn_cast<PHINode>(II); ++II)
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

        // Now loop over all of the instructions in the basic block, telling
        // dead instructions to drop their references.  This is so that the next
        // sweep over the program can safely delete dead instructions without
        // other dead instructions still refering to them.
        //
        dropReferencesOfDeadInstructionsInLiveBlock(BB);
      }
  }

  // We make changes if there are any dead blocks in the function...
  if (unsigned NumDeadBlocks = Func->size() - AliveBlocks.size()) {
    MadeChanges = true;
    NumBlockRemoved += NumDeadBlocks;
  }

  // Loop over all of the basic blocks in the function, removing control flow
  // edges to live blocks (also eliminating any entries in PHI functions in
  // referenced blocks).
  //
  for (Function::iterator BB = Func->begin(), E = Func->end(); BB != E; ++BB)
    if (!AliveBlocks.count(BB)) {
      // Remove all outgoing edges from this basic block and convert the
      // terminator into a return instruction.
      std::vector<BasicBlock*> Succs(succ_begin(BB), succ_end(BB));
      
      if (!Succs.empty()) {
        // Loop over all of the successors, removing this block from PHI node
        // entries that might be in the block...
        while (!Succs.empty()) {
          Succs.back()->removePredecessor(BB);
          Succs.pop_back();
        }
        
        // Delete the old terminator instruction...
        BB->getInstList().pop_back();
        const Type *RetTy = Func->getReturnType();
        BB->getInstList().push_back(new ReturnInst(RetTy != Type::VoidTy ?
                                           Constant::getNullValue(RetTy) : 0));
      }
    }


  // Loop over all of the basic blocks in the function, dropping references of
  // the dead basic blocks.  We must do this after the previous step to avoid
  // dropping references to PHIs which still have entries...
  //
  for (Function::iterator BB = Func->begin(), E = Func->end(); BB != E; ++BB)
    if (!AliveBlocks.count(BB))
      BB->dropAllReferences();

  // Now loop through all of the blocks and delete the dead ones.  We can safely
  // do this now because we know that there are no references to dead blocks
  // (because they have dropped all of their references...  we also remove dead
  // instructions from alive blocks.
  //
  for (Function::iterator BI = Func->begin(); BI != Func->end(); )
    if (!AliveBlocks.count(BI)) {                // Delete dead blocks...
      BI = Func->getBasicBlockList().erase(BI);
    } else {                                     // Scan alive blocks...
      for (BasicBlock::iterator II = BI->begin(); II != --BI->end(); )
        if (!LiveSet.count(II)) {             // Is this instruction alive?
          // Nope... remove the instruction from it's basic block...
          II = BI->getInstList().erase(II);
          ++NumInstRemoved;
          MadeChanges = true;
        } else {
          ++II;
        }

      ++BI;                                           // Increment iterator...
    }

  return MadeChanges;
}
