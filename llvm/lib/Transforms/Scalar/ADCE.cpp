//===- ADCE.cpp - Code to perform agressive dead code elimination ---------===//
//
// This file implements "agressive" dead code elimination.  ADCE is DCe where
// values are assumed to be dead until proven otherwise.  This is similar to 
// SCCP, except applied to the liveness of values.
//
//===----------------------------------------------------------------------===//

#include "llvm/Optimizations/DCE.h"
#include "llvm/Instruction.h"
#include "llvm/Type.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/Writer.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "Support/STLExtras.h"
#include "Support/DepthFirstIterator.h"
#include <set>
#include <algorithm>

#define DEBUG_ADCE 1

//===----------------------------------------------------------------------===//
// ADCE Class
//
// This class does all of the work of Agressive Dead Code Elimination.
// It's public interface consists of a constructor and a doADCE() method.
//
class ADCE {
  Method *M;                            // The method that we are working on...
  vector<Instruction*>   WorkList;      // Instructions that just became live
  set<Instruction*>      LiveSet;       // The set of live instructions
  bool MadeChanges;

  //===--------------------------------------------------------------------===//
  // The public interface for this class
  //
public:
  // ADCE Ctor - Save the method to operate on...
  inline ADCE(Method *m) : M(m), MadeChanges(false) {}

  // doADCE() - Run the Agressive Dead Code Elimination algorithm, returning
  // true if the method was modified.
  bool doADCE();

  //===--------------------------------------------------------------------===//
  // The implementation of this class
  //
private:
  inline void markInstructionLive(Instruction *I) {
    if (LiveSet.count(I)) return;
#ifdef DEBUG_ADCE
    cerr << "Insn Live: " << I;
#endif
    LiveSet.insert(I);
    WorkList.push_back(I);
  }

  inline void markTerminatorLive(const BasicBlock *BB) {
#ifdef DEBUG_ADCE
    cerr << "Terminat Live: " << BB->getTerminator();
#endif
    markInstructionLive((Instruction*)BB->getTerminator());
  }

  // fixupCFG - Walk the CFG in depth first order, eliminating references to 
  // dead blocks.
  //
  BasicBlock *fixupCFG(BasicBlock *Head, set<BasicBlock*> &VisitedBlocks,
		       const set<BasicBlock*> &AliveBlocks);
};



// doADCE() - Run the Agressive Dead Code Elimination algorithm, returning
// true if the method was modified.
//
bool ADCE::doADCE() {
  // Compute the control dependence graph...  Note that this has a side effect
  // on the CFG: a new return bb is added and all returns are merged here.
  //
  cfg::DominanceFrontier CDG(cfg::DominatorSet(M, true));

#ifdef DEBUG_ADCE
  cerr << "Method: " << M;
#endif

  // Iterate over all of the instructions in the method, eliminating trivially
  // dead instructions, and marking instructions live that are known to be 
  // needed.  Perform the walk in depth first order so that we avoid marking any
  // instructions live in basic blocks that are unreachable.  These blocks will
  // be eliminated later, along with the instructions inside.
  //
  for (df_iterator<Method*> BBI = df_begin(M),
                            BBE = df_end(M);
       BBI != BBE; ++BBI) {
    BasicBlock *BB = *BBI;
    for (BasicBlock::iterator II = BB->begin(), EI = BB->end(); II != EI; ) {
      Instruction *I = *II;

      if (I->hasSideEffects() || I->getOpcode() == Instruction::Ret) {
	markInstructionLive(I);
      } else {
	// Check to see if anything is trivially dead
	if (I->use_size() == 0 && I->getType() != Type::VoidTy) {
	  // Remove the instruction from it's basic block...
	  delete BB->getInstList().remove(II);
	  MadeChanges = true;
	  continue;  // Don't increment the iterator past the current slot
	}
      }

      ++II;  // Increment the inst iterator if the inst wasn't deleted
    }
  }

#ifdef DEBUG_ADCE
  cerr << "Processing work list\n";
#endif

  // AliveBlocks - Set of basic blocks that we know have instructions that are
  // alive in them...
  //
  set<BasicBlock*> AliveBlocks;

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
      cfg::DominanceFrontier::const_iterator It = CDG.find(BB);
      if (It != CDG.end()) {
	// Get the blocks that this node is control dependant on...
	const cfg::DominanceFrontier::DomSetType &CDB = It->second;
	for_each(CDB.begin(), CDB.end(),   // Mark all their terminators as live
		 bind_obj(this, &ADCE::markTerminatorLive));
      }

      // If this basic block is live, then the terminator must be as well!
      markTerminatorLive(BB);
    }

    // Loop over all of the operands of the live instruction, making sure that
    // they are known to be alive as well...
    //
    for (unsigned op = 0, End = I->getNumOperands(); op != End; ++op) {
      if (Instruction *Operand = dyn_cast<Instruction>(I->getOperand(op)))
	markInstructionLive(Operand);
    }
  }

#ifdef DEBUG_ADCE
  cerr << "Current Method: X = Live\n";
  for (Method::inst_iterator IL = M->inst_begin(); IL != M->inst_end(); ++IL) {
    if (LiveSet.count(*IL)) cerr << "X ";
    cerr << *IL;
  }
#endif

  // After the worklist is processed, recursively walk the CFG in depth first
  // order, patching up references to dead blocks...
  //
  set<BasicBlock*> VisitedBlocks;
  BasicBlock *EntryBlock = fixupCFG(M->front(), VisitedBlocks, AliveBlocks);
  if (EntryBlock && EntryBlock != M->front()) {
    if (isa<PHINode>(EntryBlock->front())) {
      // Cannot make the first block be a block with a PHI node in it! Instead,
      // strip the first basic block of the method to contain no instructions,
      // then add a simple branch to the "real" entry node...
      //
      BasicBlock *E = M->front();
      if (!isa<TerminatorInst>(E->front()) || // Check for an actual change...
	  cast<TerminatorInst>(E->front())->getNumSuccessors() != 1 ||
	  cast<TerminatorInst>(E->front())->getSuccessor(0) != EntryBlock) {
	E->getInstList().delete_all();      // Delete all instructions in block
	E->getInstList().push_back(new BranchInst(EntryBlock));
	MadeChanges = true;
      }
      AliveBlocks.insert(E);

      // Next we need to change any PHI nodes in the entry block to refer to the
      // new predecessor node...


    } else {
      // We need to move the new entry block to be the first bb of the method.
      Method::iterator EBI = find(M->begin(), M->end(), EntryBlock);
      swap(*EBI, *M->begin());  // Exchange old location with start of method
      MadeChanges = true;
    }
  }

  // Now go through and tell dead blocks to drop all of their references so they
  // can be safely deleted.
  //
  for (Method::iterator BI = M->begin(), BE = M->end(); BI != BE; ++BI) {
    BasicBlock *BB = *BI;
    if (!AliveBlocks.count(BB)) {
      BB->dropAllReferences();
    }
  }

  // Now loop through all of the blocks and delete them.  We can safely do this
  // now because we know that there are no references to dead blocks (because
  // they have dropped all of their references...
  //
  for (Method::iterator BI = M->begin(); BI != M->end();) {
    if (!AliveBlocks.count(*BI)) {
      delete M->getBasicBlocks().remove(BI);
      MadeChanges = true;
      continue;                                     // Don't increment iterator
    }
    ++BI;                                           // Increment iterator...
  }

  return MadeChanges;
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
BasicBlock *ADCE::fixupCFG(BasicBlock *BB, set<BasicBlock*> &VisitedBlocks,
			   const set<BasicBlock*> &AliveBlocks) {
  if (VisitedBlocks.count(BB)) return 0;   // Revisiting a node? No update.
  VisitedBlocks.insert(BB);                // We have now visited this node!

#ifdef DEBUG_ADCE
  cerr << "Fixing up BB: " << BB;
#endif

  if (AliveBlocks.count(BB)) {             // Is the block alive?
    // Yes it's alive: loop through and eliminate all dead instructions in block
    for (BasicBlock::iterator II = BB->begin(); II != BB->end()-1; ) {
      Instruction *I = *II;
      if (!LiveSet.count(I)) {             // Is this instruction alive?
	// Nope... remove the instruction from it's basic block...
	delete BB->getInstList().remove(II);
	MadeChanges = true;
	continue;                          // Don't increment II
      }
      ++II;
    }

    // Recursively traverse successors of this basic block.  
    BasicBlock::succ_iterator SI = BB->succ_begin(), SE = BB->succ_end();
    for (; SI != SE; ++SI) {
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
    BasicBlock::succ_iterator SI = BB->succ_begin(), SE = BB->succ_end();
    for (; SI != SE; ++SI) {
      BasicBlock *RetBB = fixupCFG(*SI, VisitedBlocks, AliveBlocks);
      if (RetBB) {
	assert(ReturnBB == 0 && "One one live child allowed!");
	ReturnBB = RetBB;
      }
    }
    return ReturnBB;                       // Return the result of traversal
  }
}



// doADCE - Execute the Agressive Dead Code Elimination Algorithm
//
bool opt::AgressiveDCE::doADCE(Method *M) {
  if (M->isExternal()) return false;
  ADCE DCE(M);
  return DCE.doADCE();
}
