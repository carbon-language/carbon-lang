//===- DCE.cpp - Code to perform dead code elimination --------------------===//
//
// This file implements dead code elimination and basic block merging.
//
// Specifically, this:
//   * removes definitions with no uses (including unused constants)
//   * removes basic blocks with no predecessors
//   * merges a basic block into its predecessor if there is only one and the
//     predecessor only has one successor.
//   * Eliminates PHI nodes for basic blocks with a single predecessor
//   * Eliminates a basic block that only contains an unconditional branch
//
// TODO: This should REALLY be recursive instead of iterative.  Right now, we 
// scan linearly through values, removing unused ones as we go.  The problem is
// that this may cause other earlier values to become unused.  To make sure that
// we get them all, we iterate until things stop changing.  Instead, when 
// removing a value, recheck all of its operands to see if they are now unused.
// Piece of cake, and more efficient as well.  
//
// Note, this is not trivial, because we have to worry about invalidating 
// iterators.  :(
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/Opt/AllOpts.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CFG.h"

using namespace cfg;

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
      //cerr << "Removing: " << *DI;
      delete Vals.remove(DI);
      Changed = true;
    } else {
      DI++;
    }
  }
  return Changed;
}

// RemoveSingularPHIs - This removes PHI nodes from basic blocks that have only
// a single predecessor.  This means that the PHI node must only have a single
// RHS value and can be eliminated.
//
// This routine is very simple because we know that PHI nodes must be the first
// things in a basic block, if they are present.
//
static bool RemoveSingularPHIs(BasicBlock *BB) {
  pred_iterator PI(pred_begin(BB));
  if (PI == pred_end(BB) || ++PI != pred_end(BB)) 
    return false;   // More than one predecessor...

  Instruction *I = BB->getInstList().front();
  if (I->getInstType() != Instruction::PHINode) return false;  // No PHI nodes

  //cerr << "Killing PHIs from " << BB;
  //cerr << "Pred #0 = " << *pred_begin(BB);

  //cerr << "Method == " << BB->getParent();

  do {
    PHINode *PN = (PHINode*)I;
    assert(PN->getOperand(2) == 0 && "PHI node should only have one value!");
    Value *V = PN->getOperand(0);

    PN->replaceAllUsesWith(V);      // Replace PHI node with its single value.
    delete BB->getInstList().remove(BB->getInstList().begin());

    I = BB->getInstList().front();
  } while (I->getInstType() == Instruction::PHINode);
	
  return true;  // Yes, we nuked at least one phi node
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

// RemovePredecessorFromBlock - This function is called when we are about
// to remove a predecessor from a basic block.  This function takes care of
// removing the predecessor from the PHI nodes in BB so that after the pred
// is removed, the number of PHI slots per bb is equal to the number of
// predecessors.
//
static void RemovePredecessorFromBlock(BasicBlock *BB, BasicBlock *Pred) {
  pred_iterator PI(pred_begin(BB)), EI(pred_end(BB));
  unsigned max_idx;

  //cerr << "RPFB: " << Pred << "From Block: " << BB;
  
  // Loop over the rest of the predecssors until we run out, or until we find
  // out that there are more than 2 predecessors.
  for (max_idx = 0; PI != EI && max_idx < 3; ++PI, ++max_idx) /*empty*/;

  // If there are exactly two predecessors, then we want to nuke the PHI nodes
  // altogether.
  bool NukePHIs = max_idx == 2;
  assert(max_idx != 0 && "PHI Node in block with 0 predecessors!?!?!");
  
  // Okay, now we know that we need to remove predecessor #pred_idx from all
  // PHI nodes.  Iterate over each PHI node fixing them up
  BasicBlock::InstListType::iterator II(BB->getInstList().begin());
  for (; (*II)->getInstType() == Instruction::PHINode; ++II) {
    PHINode *PN = (PHINode*)*II;
    PN->removeIncomingValue(BB);

    if (NukePHIs) {  // Destroy the PHI altogether??
      assert(PN->getOperand(1) == 0 && "PHI node should only have one value!");
      Value *V = PN->getOperand(0);

      PN->replaceAllUsesWith(V);      // Replace PHI node with its single value.
      delete BB->getInstList().remove(II);
    }
  }
}

// PropogatePredecessors - This gets "Succ" ready to have the predecessors from
// "BB".  This is a little tricky because "Succ" has PHI nodes, which need to
// have extra slots added to them to hold the merge edges from BB's
// predecessors.
//
// Assumption: BB is the single predecessor of Succ.
//
static void PropogatePredecessorsForPHIs(BasicBlock *BB, BasicBlock *Succ) {
  assert(BB && Succ && *pred_begin(Succ) == BB && "BB is only pred of Succ" &&
	 ++pred_begin(Succ) == pred_end(Succ));

  // If there is more than one predecessor, and there are PHI nodes in
  // the successor, then we need to add incoming edges for the PHI nodes
  pred_iterator PI(pred_begin(BB));
  for (; PI != pred_end(BB); ++PI) {
    // TODO:
  }
}

static bool DoDCEPass(Method *M) {
  Method::BasicBlocksType &BBs = M->getBasicBlocks();
  Method::BasicBlocksType::iterator BBIt, BBEnd = BBs.end();
  if (BBs.begin() == BBEnd) return false;  // Nothing to do
  bool Changed = false;

  // Loop through now and remove instructions that have no uses...
  for (BBIt = BBs.begin(); BBIt != BBEnd; BBIt++) {
    Changed |= RemoveUnusedDefs((*BBIt)->getInstList(), BasicBlockDCE());
    Changed |= RemoveSingularPHIs(*BBIt);
  }

  // Loop over all of the basic blocks (except the first one) and remove them
  // if they are unneeded...
  //
  for (BBIt = BBs.begin(), ++BBIt; BBIt != BBs.end(); ++BBIt) {
    BasicBlock *BB = *BBIt;
    assert(BB->getTerminator() && "Degenerate basic block encountered!");

#if 0
    // Remove basic blocks that have no predecessors... which are unreachable.
    if (pred_begin(BB) == pred_end(BB) &&
	!BB->hasConstantPoolReferences() && 0) {
      cerr << "Removing BB: \n" << BB;

      // Loop through all of our successors and make sure they know that one
      // of their predecessors is going away.
      for (succ_iterator SI = succ_begin(BB), EI = succ_end(BB); SI != EI; ++SI)
	RemovePredecessorFromBlock(*SI, BB);

      while (!BB->getInstList().empty()) {
	Instruction *I = BB->getInstList().front();
	// If this instruction is used, replace uses with an arbitrary
	// constant value.  Because control flow can't get here, we don't care
	// what we replace the value with.
	if (!I->use_empty()) ReplaceUsesWithConstant(I);

	// Remove the instruction from the basic block
	delete BB->getInstList().remove(BB->getInstList().begin());
      }
      delete BBs.remove(BBIt);
      --BBIt;  // remove puts use on the next block, we want the previous one
      Changed = true;
      continue;
    } 

    // Check to see if this block has no instructions and only a single 
    // successor.  If so, replace block references with successor.
    succ_iterator SI(succ_begin(BB));
    if (SI != succ_end(BB) && ++SI == succ_end(BB)) {  // One succ?
      Instruction *I = BB->getInstList().front();
      if (I->isTerminator()) {   // Terminator is the only instruction!

	if (Succ->getInstList().front()->getInstType() == Instruction::PHINode){
	  // Add entries to the PHI nodes so that the PHI nodes have the right
	  // number of entries...
	  PropogatePredecessorsForPHIs(BB, Succ);
	}

	BasicBlock *Succ = *succ_begin(BB); // There is exactly one successor
	BB->replaceAllUsesWith(Succ);
	cerr << "Killing Trivial BB: \n" << BB;

	BB = BBs.remove(BBIt);
	--BBIt; // remove puts use on the next block, we want the previous one
	
	if (BB->hasName() && !Succ->hasName())  // Transfer name if we can
	  Succ->setName(BB->getName());
	delete BB;                              // Delete basic block

	cerr << "Method after removal: \n" << M;
	Changed = true;
	continue;
      }
    }
#endif

    // Merge basic blocks into their predecessor if there is only one pred, 
    // and if there is only one successor of the predecessor. 
    pred_iterator PI(pred_begin(BB));
    if (PI != pred_end(BB) && *PI != BB &&    // Not empty?  Not same BB?
	++PI == pred_end(BB) && !BB->hasConstantPoolReferences()) {
      BasicBlock *Pred = *pred_begin(BB);
      TerminatorInst *Term = Pred->getTerminator();
      assert(Term != 0 && "malformed basic block without terminator!");

      // Does the predecessor block only have a single successor?
      succ_iterator SI(succ_begin(Pred));
      if (++SI == succ_end(Pred)) {
	//cerr << "Merging: " << BB << "into: " << Pred;

	// Delete the unconditianal branch from the predecessor...
	BasicBlock::InstListType::iterator DI = Pred->getInstList().end();
	assert(Pred->getTerminator() && 
	       "Degenerate basic block encountered!");  // Empty bb???      
	delete Pred->getInstList().remove(--DI);        // Destroy uncond branch
	
	// Move all definitions in the succecessor to the predecessor...
	while (!BB->getInstList().empty()) {
	  DI = BB->getInstList().begin();
	  Instruction *Def = BB->getInstList().remove(DI); // Remove from front
	  Pred->getInstList().push_back(Def);              // Add to end...
	}

	// Remove basic block from the method... and advance iterator to the
	// next valid block...
	BB = BBs.remove(BBIt);
	--BBIt;  // remove puts us on the NEXT bb.  We want the prev BB
	Changed = true;

	// Make all PHI nodes that refered to BB now refer to Pred as their
	// source...
	BB->replaceAllUsesWith(Pred);
	
	// Inherit predecessors name if it exists...
	if (BB->hasName() && !Pred->hasName()) Pred->setName(BB->getName());
	
	// You ARE the weakest link... goodbye
	delete BB;

	//WriteToVCG(M, "MergedInto");
      }
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
