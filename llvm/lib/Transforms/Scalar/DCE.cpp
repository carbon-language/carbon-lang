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
// TODO: This should REALLY be worklist driven instead of iterative.  Right now,
// we scan linearly through values, removing unused ones as we go.  The problem
// is that this may cause other earlier values to become unused.  To make sure
// that we get them all, we iterate until things stop changing.  Instead, when 
// removing a value, recheck all of its operands to see if they are now unused.
// Piece of cake, and more efficient as well.  
//
// Note, this is not trivial, because we have to worry about invalidating 
// iterators.  :(
//
//===----------------------------------------------------------------------===//

#include "llvm/Optimizations/DCE.h"
#include "llvm/Support/STLExtras.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CFG.h"
#include <algorithm>

using namespace cfg;

struct ConstPoolDCE { 
  enum { EndOffs = 0 };
  static bool isDCEable(const ConstPoolVal *CPV) {
    // TODO: The bytecode writer requires that all used types are in the
    // constant pool for the current method.  This is messy and is really
    // irritating. FIXME
    return CPV->getType() != Type::TypeTy;  // Don't DCE Type plane constants!
  }
};

struct BasicBlockDCE {
  enum { EndOffs = 1 };
  static bool isDCEable(const Instruction *I) {
    return !I->hasSideEffects();
  }
};


template<class Container, class DCEController>
static bool RemoveUnusedDefs(Container &Vals, DCEController DCEControl) {
  bool Changed = false;
  int Offset = DCEController::EndOffs;

  for (typename Container::iterator DI = Vals.begin(); 
       DI != Vals.end()-Offset; ) {
    // Look for un"used" definitions...
    if ((*DI)->use_empty() && DCEController::isDCEable(*DI)) {
      // Bye bye
      //cerr << "Removing: " << *DI;
      delete Vals.remove(DI);
      Changed = true;
    } else {
      ++DI;
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

  Instruction *I = BB->front();
  if (!I->isPHINode()) return false;  // No PHI nodes

  //cerr << "Killing PHIs from " << BB;
  //cerr << "Pred #0 = " << *pred_begin(BB);

  //cerr << "Method == " << BB->getParent();

  do {
    PHINode *PN = (PHINode*)I;
    assert(PN->getNumOperands() == 2 && "PHI node should only have one value!");
    Value *V = PN->getOperand(0);

    PN->replaceAllUsesWith(V);      // Replace PHI node with its single value.
    delete BB->getInstList().remove(BB->begin());

    I = BB->front();
  } while (I->isPHINode());
	
  return true;  // Yes, we nuked at least one phi node
}

bool opt::DoRemoveUnusedConstants(SymTabValue *S) {
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

// PropogatePredecessors - This gets "Succ" ready to have the predecessors from
// "BB".  This is a little tricky because "Succ" has PHI nodes, which need to
// have extra slots added to them to hold the merge edges from BB's
// predecessors.
//
// Assumption: BB is the single predecessor of Succ.
//
static void PropogatePredecessorsForPHIs(BasicBlock *BB, BasicBlock *Succ) {
  assert(Succ->front()->isPHINode() && "Only works on PHId BBs!");

  // If there is more than one predecessor, and there are PHI nodes in
  // the successor, then we need to add incoming edges for the PHI nodes
  //
  const vector<BasicBlock*> BBPreds(pred_begin(BB), pred_end(BB));

  BasicBlock::iterator I = Succ->begin();
  do {                     // Loop over all of the PHI nodes in the successor BB
    PHINode *PN = (PHINode*)*I;
    Value *OldVal = PN->removeIncomingValue(BB);
    assert(OldVal && "No entry in PHI for Pred BB!");

    for (vector<BasicBlock*>::const_iterator PredI = BBPreds.begin(), 
	   End = BBPreds.end(); PredI != End; ++PredI) {
      // Add an incoming value for each of the new incoming values...
      PN->addIncoming(OldVal, *PredI);
    }

    ++I;
  } while ((*I)->isPHINode());
}


// SimplifyCFG - This function is used to do simplification of a CFG.  For
// example, it adjusts branches to branches to eliminate the extra hop, it
// eliminates unreachable basic blocks, and does other "peephole" optimization
// of the CFG.  It returns true if a modification was made, and returns an 
// iterator that designates the first element remaining after the block that
// was deleted.
//
// WARNING:  The entry node of a method may not be simplified.
//
bool opt::SimplifyCFG(Method::iterator &BBIt) {
  assert(*BBIt && (*BBIt)->getParent() && "Block not embedded in method!");
  BasicBlock *BB = *BBIt;
  Method *M = BB->getParent();
  assert(BB->getTerminator() && "Degenerate basic block encountered!");
  assert(BB->getParent()->front() != BB && "Can't Simplify entry block!");

  // Remove basic blocks that have no predecessors... which are unreachable.
  if (pred_begin(BB) == pred_end(BB) &&
      !BB->hasConstantPoolReferences()) {
    //cerr << "Removing BB: \n" << BB;

    // Loop through all of our successors and make sure they know that one
    // of their predecessors is going away.
    for_each(succ_begin(BB), succ_end(BB),
	     std::bind2nd(std::mem_fun(&BasicBlock::removePredecessor), BB));

    while (!BB->empty()) {
      Instruction *I = BB->back();
      // If this instruction is used, replace uses with an arbitrary
      // constant value.  Because control flow can't get here, we don't care
      // what we replace the value with.  Note that since this block is 
      // unreachable, and all values contained within it must dominate their
      // uses, that all uses will eventually be removed.
      if (!I->use_empty()) ReplaceUsesWithConstant(I);
      
      // Remove the instruction from the basic block
      delete BB->getInstList().pop_back();
    }
    delete M->getBasicBlocks().remove(BBIt);
    return true;
  } 

  // Check to see if this block has no instructions and only a single 
  // successor.  If so, replace block references with successor.
  succ_iterator SI(succ_begin(BB));
  if (SI != succ_end(BB) && ++SI == succ_end(BB)) {  // One succ?
    Instruction *I = BB->front();
    if (I->isTerminator()) {   // Terminator is the only instruction!
      BasicBlock *Succ = *succ_begin(BB); // There is exactly one successor
      //cerr << "Killing Trivial BB: \n" << BB;
      
      if (Succ != BB) {   // Arg, don't hurt infinite loops!
	if (Succ->front()->isPHINode()) {
	  // If our successor has PHI nodes, then we need to update them to
	  // include entries for BB's predecessors, not for BB itself.
	  //
	  PropogatePredecessorsForPHIs(BB, Succ);
	}
	
	BB->replaceAllUsesWith(Succ);
	BB = M->getBasicBlocks().remove(BBIt);
	
	if (BB->hasName() && !Succ->hasName())  // Transfer name if we can
	  Succ->setName(BB->getName());
	delete BB;                              // Delete basic block
	
	//cerr << "Method after removal: \n" << M;
	return true;
      }
    }
  }

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
      BasicBlock::iterator DI = Pred->end();
      assert(Pred->getTerminator() && 
	     "Degenerate basic block encountered!");  // Empty bb???      
      delete Pred->getInstList().remove(--DI);        // Destroy uncond branch
      
      // Move all definitions in the succecessor to the predecessor...
      while (!BB->empty()) {
	DI = BB->begin();
	Instruction *Def = BB->getInstList().remove(DI); // Remove from front
	Pred->getInstList().push_back(Def);              // Add to end...
      }
      
      // Remove basic block from the method... and advance iterator to the
      // next valid block...
      BB = M->getBasicBlocks().remove(BBIt);

      // Make all PHI nodes that refered to BB now refer to Pred as their
      // source...
      BB->replaceAllUsesWith(Pred);
      
      // Inherit predecessors name if it exists...
      if (BB->hasName() && !Pred->hasName()) Pred->setName(BB->getName());
      
      delete BB; // You ARE the weakest link... goodbye
      return true;
    }
  }
  
  return false;
}

static bool DoDCEPass(Method *M) {
  Method::iterator BBIt, BBEnd = M->end();
  if (M->begin() == BBEnd) return false;  // Nothing to do
  bool Changed = false;

  // Loop through now and remove instructions that have no uses...
  for (BBIt = M->begin(); BBIt != BBEnd; ++BBIt) {
    Changed |= RemoveUnusedDefs((*BBIt)->getInstList(), BasicBlockDCE());
    Changed |= RemoveSingularPHIs(*BBIt);
  }

  // Loop over all of the basic blocks (except the first one) and remove them
  // if they are unneeded...
  //
  for (BBIt = M->begin(), ++BBIt; BBIt != M->end(); ) {
    if (opt::SimplifyCFG(BBIt)) {
      Changed = true;
    } else {
      ++BBIt;
    }
  }

  // Remove unused constants
  return Changed | opt::DoRemoveUnusedConstants(M);
}


// It is possible that we may require multiple passes over the code to fully
// eliminate dead code.  Iterate until we are done.
//
bool opt::DoDeadCodeElimination(Method *M) {
  bool Changed = false;
  while (DoDCEPass(M)) Changed = true;
  return Changed;
}

bool opt::DoDeadCodeElimination(Module *C) { 
  bool Val = C->reduceApply(DoDeadCodeElimination);

  while (DoRemoveUnusedConstants(C)) Val = true;
  return Val;
}
