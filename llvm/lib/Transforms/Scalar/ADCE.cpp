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
#include <set>

#include "llvm/Assembly/Writer.h"

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
    case Instruction::Call:
    case Instruction::Store:
      markInstructionLive(I);
      break;
    default:
      if (I->getType() == Type::VoidTy) {
	markInstructionLive(I);   // Catches terminators and friends
      } else {
	if (I->use_size() == 0) { // Check to see if anything is trivially dead
	  // Remove the instruction from it's basic block...
	  BasicBlock *BB = I->getParent();
	  delete BB->getInstList().remove(II.getInstructionIterator());

	  // Make sure to sync up the iterator again...
	  II.resyncInstructionIterator();
	  continue;  // Don't increment the iterator past the current slot
	}
      }
    }

    ++II;  // Increment the iterator
  }


  cerr << "Processing work list\n";

  // Process the work list of instructions that just became live... if they
  // became live, then that means that all of their operands are neccesary as
  // well... make them live as well.
  //
  while (!WorkList.empty()) {
    Instruction *I = WorkList.back();
    WorkList.pop_back();

    for (unsigned op = 0; Value *Op = I->getOperand(op); ++op) {
      Instruction *Operand = Op->castInstruction();
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
