//===- ADCE.cpp - Code to perform aggressive dead code elimination --------===//
//
// This file implements "aggressive" dead code elimination.  ADCE is DCe where
// values are assumed to be dead until proven otherwise.  This is similar to 
// SCCP, except applied to the liveness of values.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Type.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/Writer.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/Support/CFG.h"
#include "Support/STLExtras.h"
#include "Support/DepthFirstIterator.h"
#include "Support/StatisticReporter.h"
#include <algorithm>
#include <iostream>
using std::cerr;

namespace {

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
  bool MadeChanges;

  //===--------------------------------------------------------------------===//
  // The public interface for this class
  //
public:
  const char *getPassName() const { return "Aggressive Dead Code Elimination"; }
  
  // doADCE - Execute the Aggressive Dead Code Elimination Algorithm
  //
  virtual bool runOnFunction(Function *F) {
    Func = F; MadeChanges = false;
    doADCE(getAnalysis<DominanceFrontier>(DominanceFrontier::PostDomID));
    assert(WorkList.empty());
    LiveSet.clear();
    return MadeChanges;
  }
  // getAnalysisUsage - We require post dominance frontiers (aka Control
  // Dependence Graph)
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired(DominanceFrontier::PostDomID);
  }


  //===--------------------------------------------------------------------===//
  // The implementation of this class
  //
private:
  // doADCE() - Run the Aggressive Dead Code Elimination algorithm, returning
  // true if the function was modified.
  //
  void doADCE(DominanceFrontier &CDG);

  inline void markInstructionLive(Instruction *I) {
    if (LiveSet.count(I)) return;
    DEBUG(cerr << "Insn Live: " << I);
    LiveSet.insert(I);
    WorkList.push_back(I);
  }

  inline void markTerminatorLive(const BasicBlock *BB) {
    DEBUG(cerr << "Terminat Live: " << BB->getTerminator());
    markInstructionLive((Instruction*)BB->getTerminator());
  }

  // fixupCFG - Walk the CFG in depth first order, eliminating references to 
  // dead blocks.
  //
  BasicBlock *fixupCFG(BasicBlock *Head, std::set<BasicBlock*> &VisitedBlocks,
		       const std::set<BasicBlock*> &AliveBlocks);
};

} // End of anonymous namespace

Pass *createAggressiveDCEPass() {
  return new ADCE();
}


// doADCE() - Run the Aggressive Dead Code Elimination algorithm, returning
// true if the function was modified.
//
void ADCE::doADCE(DominanceFrontier &CDG) {
  DEBUG(cerr << "Function: " << Func);

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
      Instruction *I = *II;

      if (I->hasSideEffects() || I->getOpcode() == Instruction::Ret) {
	markInstructionLive(I);
        ++II;  // Increment the inst iterator if the inst wasn't deleted
      } else if (isInstructionTriviallyDead(I)) {
        // Remove the instruction from it's basic block...
        delete BB->getInstList().remove(II);
        MadeChanges = true;
      } else {
        ++II;  // Increment the inst iterator if the inst wasn't deleted
      }
    }
  }

  DEBUG(cerr << "Processing work list\n");

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
    if (AliveBlocks.count(BB) == 0) {   // Basic block not alive yet...
      // Mark the basic block as being newly ALIVE... and mark all branches that
      // this block is control dependant on as being alive also...
      //
      AliveBlocks.insert(BB);   // Block is now ALIVE!
      DominanceFrontier::const_iterator It = CDG.find(BB);
      if (It != CDG.end()) {
	// Get the blocks that this node is control dependant on...
	const DominanceFrontier::DomSetType &CDB = It->second;
	for_each(CDB.begin(), CDB.end(),   // Mark all their terminators as live
		 bind_obj(this, &ADCE::markTerminatorLive));
      }

      // If this basic block is live, then the terminator must be as well!
      markTerminatorLive(BB);
    }

    // Loop over all of the operands of the live instruction, making sure that
    // they are known to be alive as well...
    //
    for (unsigned op = 0, End = I->getNumOperands(); op != End; ++op)
      if (Instruction *Operand = dyn_cast<Instruction>(I->getOperand(op)))
	markInstructionLive(Operand);
  }

  if (DebugFlag) {
    cerr << "Current Function: X = Live\n";
    for (Function::iterator I = Func->begin(), E = Func->end(); I != E; ++I)
      for (BasicBlock::iterator BI = (*I)->begin(), BE = (*I)->end();
           BI != BE; ++BI) {
        if (LiveSet.count(*BI)) cerr << "X ";
        cerr << *BI;
      }
  }

  // After the worklist is processed, recursively walk the CFG in depth first
  // order, patching up references to dead blocks...
  //
  std::set<BasicBlock*> VisitedBlocks;
  BasicBlock *EntryBlock = fixupCFG(Func->front(), VisitedBlocks, AliveBlocks);

  // Now go through and tell dead blocks to drop all of their references so they
  // can be safely deleted.  Also, as we are doing so, if the block has
  // successors that are still live (and that have PHI nodes in them), remove
  // the entry for this block from the phi nodes.
  //
  for (Function::iterator BI = Func->begin(), BE = Func->end(); BI != BE; ++BI){
    BasicBlock *BB = *BI;
    if (!AliveBlocks.count(BB)) {
      // Remove entries from successors PHI nodes if they are still alive...
      for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI)
        if (AliveBlocks.count(*SI)) {  // Only if the successor is alive...
          BasicBlock *Succ = *SI;
          for (BasicBlock::iterator I = Succ->begin();// Loop over all PHI nodes
               PHINode *PN = dyn_cast<PHINode>(*I); ++I)
            PN->removeIncomingValue(BB);         // Remove value for this block
        }

      BB->dropAllReferences();
    }
  }

  cerr << "Before Deleting Blocks: " << Func;

  // Now loop through all of the blocks and delete them.  We can safely do this
  // now because we know that there are no references to dead blocks (because
  // they have dropped all of their references...
  //
  for (Function::iterator BI = Func->begin(); BI != Func->end();) {
    if (!AliveBlocks.count(*BI)) {
      delete Func->getBasicBlocks().remove(BI);
      MadeChanges = true;
      continue;                                     // Don't increment iterator
    }
    ++BI;                                           // Increment iterator...
  }

  if (EntryBlock && EntryBlock != Func->front()) {
    // We need to move the new entry block to be the first bb of the function
    Function::iterator EBI = find(Func->begin(), Func->end(), EntryBlock);
    std::swap(*EBI, *Func->begin()); // Exchange old location with start of fn
  }

  while (PHINode *PN = dyn_cast<PHINode>(EntryBlock->front())) {
    assert(PN->getNumIncomingValues() == 1 &&
           "Can only have a single incoming value at this point...");
    // The incoming value must be outside of the scope of the function, a
    // global variable, constant or parameter maybe...
    //
    PN->replaceAllUsesWith(PN->getIncomingValue(0));
    
    // Nuke the phi node...
    delete EntryBlock->getInstList().remove(EntryBlock->begin());
  }
}


// fixupCFG - Walk the CFG in depth first order, eliminating references to 
// dead blocks:
//  If the BB is alive (in AliveBlocks):
//   1. Eliminate all dead instructions in the BB
//   2. Recursively traverse all of the successors of the BB:
//      - If the returned successor is non-null, update our terminator to
//         reference the returned BB
//   3. Return 0 (no update needed)
//
//  If the BB is dead (not in AliveBlocks):
//   1. Add the BB to the dead set
//   2. Recursively traverse all of the successors of the block:
//      - Only one shall return a nonnull value (or else this block should have
//        been in the alive set).
//   3. Return the nonnull child, or 0 if no non-null children.
//
BasicBlock *ADCE::fixupCFG(BasicBlock *BB, std::set<BasicBlock*> &VisitedBlocks,
			   const std::set<BasicBlock*> &AliveBlocks) {
  if (VisitedBlocks.count(BB)) return 0;   // Revisiting a node? No update.
  VisitedBlocks.insert(BB);                // We have now visited this node!

  DEBUG(cerr << "Fixing up BB: " << BB);

  if (AliveBlocks.count(BB)) {             // Is the block alive?
    // Yes it's alive: loop through and eliminate all dead instructions in block
    for (BasicBlock::iterator II = BB->begin(); II != BB->end()-1; )
      if (!LiveSet.count(*II)) {             // Is this instruction alive?
	// Nope... remove the instruction from it's basic block...
	delete BB->getInstList().remove(II);
	MadeChanges = true;
      } else {
        ++II;
      }

    // Recursively traverse successors of this basic block.  
    for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI) {
      BasicBlock *Succ = *SI;
      BasicBlock *Repl = fixupCFG(Succ, VisitedBlocks, AliveBlocks);
      if (Repl && Repl != Succ) {          // We have to replace the successor
	Succ->replaceAllUsesWith(Repl);
	MadeChanges = true;
      }
    }
    return BB;
  } else {                                 // Otherwise the block is dead...
    BasicBlock *ReturnBB = 0;              // Default to nothing live down here
    
    // Recursively traverse successors of this basic block.  
    for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI) {
      BasicBlock *RetBB = fixupCFG(*SI, VisitedBlocks, AliveBlocks);
      if (RetBB) {
	assert(ReturnBB == 0 && "At most one live child allowed!");
	ReturnBB = RetBB;
      }
    }
    return ReturnBB;                       // Return the result of traversal
  }
}

