//===- DCE.cpp - Code to perform dead code elimination --------------------===//
//
// This file implements dead code elimination and basic block merging.
//
// Specifically, this:
//   * removes definitions with no uses (including unused constants)
//   * removes basic blocks with no predecessors
//   * merges a basic block into its predecessor if there is only one and the
//     predecessor only has one successor.
//
// TODO: This should REALLY be recursive instead of iterative.  Right now, we 
// scan linearly through values, removing unused ones as we go.  The problem is
// that this may cause other earlier values to become unused.  To make sure that
// we get them all, we iterate until things stop changing.  Instead, when 
// removing a value, recheck all of its operands to see if they are now unused.
// Piece of cake, and more efficient as well.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"
#include "llvm/iTerminators.h"
#include "llvm/Opt/AllOpts.h"

struct ConstPoolDCE { 
  enum { EndOffs = 0 };
  static bool isDCEable(const Value *) { return true; } 
};

struct BasicBlockDCE {
  enum { EndOffs = 1 };
  static bool isDCEable(const Instruction *I) {
    return !I->hasSideEffects();
  }
};

template<class ValueSubclass, class ItemParentType, class DCEController>
static bool RemoveUnusedDefs(ValueHolder<ValueSubclass, ItemParentType> &Vals, 
			     DCEController DCEControl) {
  bool Changed = false;
  typedef ValueHolder<ValueSubclass, ItemParentType> Container;

  int Offset = DCEController::EndOffs;
  for (Container::iterator DI = Vals.begin(); DI != Vals.end()-Offset; ) {
    // Look for un"used" definitions...
    if ((*DI)->use_empty() && DCEController::isDCEable(*DI)) {
      // Bye bye
      delete Vals.remove(DI);
      Changed = true;
    } else {
      DI++;
    }
  }
  return Changed;
}


bool DoRemoveUnusedConstants(SymTabValue *S) {
  bool Changed = false;
  ConstantPool &CP = S->getConstantPool();
  for (ConstantPool::plane_iterator PI = CP.begin(); PI != CP.end(); ++PI)
    Changed |= RemoveUnusedDefs(**PI, ConstPoolDCE());
  return Changed;
}


static void ReplaceUsesWithConstant(Instruction *I) {
  // Get the method level constant pool
  ConstantPool &CP = I->getParent()->getParent()->getConstantPool();

  ConstPoolVal *CPV = 0;
  ConstantPool::PlaneType *P;
  if (!CP.getPlane(I->getType(), P)) {  // Does plane exist?
    // Yes, is it empty?
    if (!P->empty()) CPV = P->front();
  }

  if (CPV == 0) { // We don't have an existing constant to reuse.  Just add one.
    CPV = ConstPoolVal::getNullConstant(I->getType());  // Create a new constant

    // Add the new value to the constant pool...
    CP.insert(CPV);
  }
  
  // Make all users of this instruction reference the constant instead
  I->replaceAllUsesWith(CPV);
}

static bool DoDCEPass(Method *M) {
  Method::BasicBlocksType::iterator BBIt;
  Method::BasicBlocksType &BBs = M->getBasicBlocks();
  bool Changed = false;

  // Loop through now and remove instructions that have no uses...
  for (BBIt = BBs.begin(); BBIt != BBs.end(); BBIt++)
    Changed |= RemoveUnusedDefs((*BBIt)->getInstList(), BasicBlockDCE());

  // Scan through and remove basic blocks that have no predecessors (except,
  // of course, the first one.  :)  (so skip first block)
  //
  for (BBIt = BBs.begin(), ++BBIt; BBIt != BBs.end(); BBIt++) {
    BasicBlock *BB = *BBIt;
    assert(BB->getTerminator() && 
	   "Degenerate basic block encountered!");  // Empty bb???

    if (BB->pred_begin() == BB->pred_end() &&
	!BB->hasConstantPoolReferences()) {

      while (!BB->getInstList().empty()) {
	Instruction *I = BB->getInstList().front();
	// If this instruction is used, replace uses with an arbitrary
	// constant value.  Because control flow can't get here, we don't care
	// what we replace the value with.
	if (!I->use_empty()) ReplaceUsesWithConstant(I);

	// Remove the instruction from the basic block
	BasicBlock::InstListType::iterator f = BB->getInstList().begin();
	delete BB->getInstList().remove(f);
      }

      delete BBs.remove(BBIt);
      ++BBIt;  // remove puts use on the previous block, we want the next one
      Changed = true;
    }
  }

  // Loop through an merge basic blocks into their predecessor if there is only
  // one, and if there is only one successor of the predecessor.
  //
  for (BBIt = BBs.begin(); BBIt != BBs.end(); BBIt++) {
    BasicBlock *BB = *BBIt;

    // Is there exactly one predecessor to this block?
    BasicBlock::pred_iterator PI(BB->pred_begin());
    if (PI != BB->pred_end() && ++PI == BB->pred_end() && 
	!BB->hasConstantPoolReferences()) {
      BasicBlock *Pred = *BB->pred_begin();
      TerminatorInst *Term = Pred->getTerminator();
      if (Term == 0) continue; // Err... malformed basic block!

      // Is it an unconditional branch?
      if (Term->getInstType() != Instruction::Br ||
          !((BranchInst*)Term)->isUnconditional())
        continue;  // Nope, maybe next time...

      Changed = true;

      // Make all branches to the predecessor now point to the successor...
      Pred->replaceAllUsesWith(BB);

      // Move all definitions in the predecessor to the successor...
      BasicBlock::InstListType::iterator DI = Pred->getInstList().end();
      assert(Pred->getTerminator() && 
	     "Degenerate basic block encountered!");  // Empty bb???      
      delete Pred->getInstList().remove(--DI); // Remove terminator
      
      while (Pred->getInstList().begin() != (DI = Pred->getInstList().end())) {
        Instruction *Def = Pred->getInstList().remove(--DI); // Remove from end
        BB->getInstList().push_front(Def);                   // Add to front...
      }

      // Remove basic block from the method...
      BBs.remove(Pred);

      // Always inherit predecessors name if it exists...
      if (Pred->hasName()) BB->setName(Pred->getName());

      // So long you waste of a basic block you...
      delete Pred;
    }
  }

  // Remove unused constants
  Changed |= DoRemoveUnusedConstants(M);
  return Changed;
}


// It is possible that we may require multiple passes over the code to fully
// eliminate dead code.  Iterate until we are done.
//
bool DoDeadCodeElimination(Method *M) {
  bool Changed = false;
  while (DoDCEPass(M)) Changed = true;
  return Changed;
}

bool DoDeadCodeElimination(Module *C) { 
  bool Val = ApplyOptToAllMethods(C, DoDeadCodeElimination);
  while (DoRemoveUnusedConstants(C)) Val = true;
  return Val;
}
