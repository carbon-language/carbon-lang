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
#include "llvm/Tools/STLExtras.h"
#include "llvm/Analysis/Writer.h"
#include <set>
#include <algorithm>

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

  //===--------------------------------------------------------------------===//
  // The public interface for this class
  //
public:
  // ADCE Ctor - Save the method to operate on...
  inline ADCE(Method *m) : M(m) {}

  // doADCE() - Run the Agressive Dead Code Elimination algorithm, returning
  // true if the method was modified.
  bool doADCE();

  //===--------------------------------------------------------------------===//
  // The implementation of this class
  //
private:
  inline void markInstructionLive(Instruction *I) {
    if (LiveSet.count(I)) return;
    cerr << "Insn Live: " << I;
    LiveSet.insert(I);
    WorkList.push_back(I);
  }

  inline void markTerminatorLive(const BasicBlock *BB) {
    cerr << "Marking Term Live\n";
    markInstructionLive((Instruction*)BB->back());
  }
};



// doADCE() - Run the Agressive Dead Code Elimination algorithm, returning
// true if the method was modified.
//
bool ADCE::doADCE() {
  // Iterate over all of the instructions in the method, eliminating trivially
  // dead instructions, and marking instructions live that are known to be 
  // needed.
  //
  for (Method::inst_iterator II = M->inst_begin(); II != M->inst_end(); ) {
    Instruction *I = *II;
    switch (I->getInstType()) {
    case Instruction::Ret:
    case Instruction::Call:
    case Instruction::Store:
      markInstructionLive(I);
      break;
    default:
      // Check to see if anything is trivially dead
      if (I->use_size() == 0 && I->getType() != Type::VoidTy) {
	// Remove the instruction from it's basic block...
	BasicBlock *BB = I->getParent();
	delete BB->getInstList().remove(II.getInstructionIterator());
	
	// Make sure to sync up the iterator again...
	II.resyncInstructionIterator();
	continue;  // Don't increment the iterator past the current slot
      }
    }

    ++II;  // Increment the iterator
  }

  // Compute the control dependence graph...
  cfg::DominanceFrontier CDG(cfg::DominatorSet(M, true));

  cerr << "Processing work list\n";

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
    }

    for (unsigned op = 0, End = I->getNumOperands(); op != End; ++op) {
      Instruction *Operand = I->getOperand(op)->castInstruction();
      if (Operand) markInstructionLive(Operand);
    }
  }

  // After the worklist is processed, loop through the instructions again,
  // removing any that are not live... by the definition of the LiveSet.
  //
  for (Method::inst_iterator II = M->inst_begin(); II != M->inst_end(); ) {
    Instruction *I = *II;
    if (!LiveSet.count(I)) {
      cerr << "Instruction Dead: " << I;
    }

    ++II;  // Increment the iterator
  }

  return false;
}


// DoADCE - Execute the Agressive Dead Code Elimination Algorithm
//
bool opt::DoADCE(Method *M) {
  ADCE DCE(M);
  return DCE.doADCE();
}
