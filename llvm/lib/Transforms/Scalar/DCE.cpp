//===- DCE.cpp - Code to perform dead code elimination --------------------===//
//
// This file implements dead code elimination and basic block merging.
//
// Specifically, this:
//   * removes definitions with no uses
//   * removes basic blocks with no predecessors
//   * merges a basic block into its predecessor if there is only one and the
//     predecessor only has one successor.
//   * Eliminates PHI nodes for basic blocks with a single predecessor
//   * Eliminates a basic block that only contains an unconditional branch
//   * Eliminates function prototypes that are not referenced
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

#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Module.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/ConstantVals.h"
#include "llvm/Support/CFG.h"
#include "llvm/Pass.h"
#include "Support/STLExtras.h"
#include <algorithm>

// dceInstruction - Inspect the instruction at *BBI and figure out if it's
// [trivially] dead.  If so, remove the instruction and update the iterator
// to point to the instruction that immediately succeeded the original
// instruction.
//
bool dceInstruction(BasicBlock::InstListType &BBIL,
                    BasicBlock::iterator &BBI) {
  // Look for un"used" definitions...
  if ((*BBI)->use_empty() && !(*BBI)->hasSideEffects() && 
      !isa<TerminatorInst>(*BBI)) {
    delete BBIL.remove(BBI);   // Bye bye
    return true;
  }
  return false;
}

static inline bool RemoveUnusedDefs(BasicBlock::InstListType &Vals) {
  bool Changed = false;
  for (BasicBlock::InstListType::iterator DI = Vals.begin(); 
       DI != Vals.end(); )
    if (dceInstruction(Vals, DI))
      Changed = true;
    else
      ++DI;
  return Changed;
}

struct DeadInstElimination : public BasicBlockPass {
  virtual bool runOnBasicBlock(BasicBlock *BB) {
    return RemoveUnusedDefs(BB->getInstList());
  }
};

Pass *createDeadInstEliminationPass() {
  return new DeadInstElimination();
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
  if (!isa<PHINode>(I)) return false;  // No PHI nodes

  //cerr << "Killing PHIs from " << BB;
  //cerr << "Pred #0 = " << *pred_begin(BB);

  //cerr << "Function == " << BB->getParent();

  do {
    PHINode *PN = cast<PHINode>(I);
    assert(PN->getNumOperands() == 2 && "PHI node should only have one value!");
    Value *V = PN->getOperand(0);

    PN->replaceAllUsesWith(V);      // Replace PHI node with its single value.
    delete BB->getInstList().remove(BB->begin());

    I = BB->front();
  } while (isa<PHINode>(I));
	
  return true;  // Yes, we nuked at least one phi node
}

static void ReplaceUsesWithConstant(Instruction *I) {
  // Make all users of this instruction reference the constant instead
  I->replaceAllUsesWith(Constant::getNullValue(I->getType()));
}

// PropogatePredecessors - This gets "Succ" ready to have the predecessors from
// "BB".  This is a little tricky because "Succ" has PHI nodes, which need to
// have extra slots added to them to hold the merge edges from BB's
// predecessors.  This function returns true (failure) if the Succ BB already
// has a predecessor that is a predecessor of BB.
//
// Assumption: Succ is the single successor for BB.
//
static bool PropogatePredecessorsForPHIs(BasicBlock *BB, BasicBlock *Succ) {
  assert(*succ_begin(BB) == Succ && "Succ is not successor of BB!");
  assert(isa<PHINode>(Succ->front()) && "Only works on PHId BBs!");

  // If there is more than one predecessor, and there are PHI nodes in
  // the successor, then we need to add incoming edges for the PHI nodes
  //
  const std::vector<BasicBlock*> BBPreds(pred_begin(BB), pred_end(BB));

  // Check to see if one of the predecessors of BB is already a predecessor of
  // Succ.  If so, we cannot do the transformation!
  //
  for (pred_iterator PI = pred_begin(Succ), PE = pred_end(Succ);
       PI != PE; ++PI) {
    if (find(BBPreds.begin(), BBPreds.end(), *PI) != BBPreds.end())
      return true;
  }

  BasicBlock::iterator I = Succ->begin();
  do {                     // Loop over all of the PHI nodes in the successor BB
    PHINode *PN = cast<PHINode>(*I);
    Value *OldVal = PN->removeIncomingValue(BB);
    assert(OldVal && "No entry in PHI for Pred BB!");

    for (std::vector<BasicBlock*>::const_iterator PredI = BBPreds.begin(), 
	   End = BBPreds.end(); PredI != End; ++PredI) {
      // Add an incoming value for each of the new incoming values...
      PN->addIncoming(OldVal, *PredI);
    }

    ++I;
  } while (isa<PHINode>(*I));
  return false;
}


// SimplifyCFG - This function is used to do simplification of a CFG.  For
// example, it adjusts branches to branches to eliminate the extra hop, it
// eliminates unreachable basic blocks, and does other "peephole" optimization
// of the CFG.  It returns true if a modification was made, and returns an 
// iterator that designates the first element remaining after the block that
// was deleted.
//
// WARNING:  The entry node of a function may not be simplified.
//
bool SimplifyCFG(Function::iterator &BBIt) {
  BasicBlock *BB = *BBIt;
  Function *M = BB->getParent();

  assert(BB && BB->getParent() && "Block not embedded in function!");
  assert(BB->getTerminator() && "Degenerate basic block encountered!");
  assert(BB->getParent()->front() != BB && "Can't Simplify entry block!");


  // Remove basic blocks that have no predecessors... which are unreachable.
  if (pred_begin(BB) == pred_end(BB) &&
      !BB->hasConstantReferences()) {
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
    if (BB->front()->isTerminator()) {   // Terminator is the only instruction!
      BasicBlock *Succ = *succ_begin(BB); // There is exactly one successor
      //cerr << "Killing Trivial BB: \n" << BB;
      
      if (Succ != BB) {   // Arg, don't hurt infinite loops!
        // If our successor has PHI nodes, then we need to update them to
        // include entries for BB's predecessors, not for BB itself.
        // Be careful though, if this transformation fails (returns true) then
        // we cannot do this transformation!
        //
	if (!isa<PHINode>(Succ->front()) ||
            !PropogatePredecessorsForPHIs(BB, Succ)) {
          
          BB->replaceAllUsesWith(Succ);
          BB = M->getBasicBlocks().remove(BBIt);
	
          if (BB->hasName() && !Succ->hasName())  // Transfer name if we can
            Succ->setName(BB->getName());
          delete BB;                              // Delete basic block
          
          //cerr << "Function after removal: \n" << M;
          return true;
	}
      }
    }
  }

  // Merge basic blocks into their predecessor if there is only one pred, 
  // and if there is only one successor of the predecessor. 
  pred_iterator PI(pred_begin(BB));
  if (PI != pred_end(BB) && *PI != BB &&    // Not empty?  Not same BB?
      ++PI == pred_end(BB) && !BB->hasConstantReferences()) {
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
      
      // Remove basic block from the function... and advance iterator to the
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

static bool DoDCEPass(Function *F) {
  Function::iterator BBIt, BBEnd = F->end();
  if (F->begin() == BBEnd) return false;  // Nothing to do
  bool Changed = false;

  // Loop through now and remove instructions that have no uses...
  for (BBIt = F->begin(); BBIt != BBEnd; ++BBIt) {
    Changed |= RemoveUnusedDefs((*BBIt)->getInstList());
    Changed |= RemoveSingularPHIs(*BBIt);
  }

  // Loop over all of the basic blocks (except the first one) and remove them
  // if they are unneeded...
  //
  for (BBIt = F->begin(), ++BBIt; BBIt != F->end(); ) {
    if (SimplifyCFG(BBIt)) {
      Changed = true;
    } else {
      ++BBIt;
    }
  }

  return Changed;
}

// Remove unused global values - This removes unused global values of no
// possible value.  This currently includes unused function prototypes and
// unitialized global variables.
//
static bool RemoveUnusedGlobalValues(Module *Mod) {
  bool Changed = false;

  for (Module::iterator MI = Mod->begin(); MI != Mod->end(); ) {
    Function *Meth = *MI;
    if (Meth->isExternal() && Meth->use_size() == 0) {
      // No references to prototype?
      //cerr << "Removing function proto: " << Meth->getName() << endl;
      delete Mod->getFunctionList().remove(MI);  // Remove prototype
      // Remove moves iterator to point to the next one automatically
      Changed = true;
    } else {
      ++MI;                                    // Skip prototype in use.
    }
  }

  for (Module::giterator GI = Mod->gbegin(); GI != Mod->gend(); ) {
    GlobalVariable *GV = *GI;
    if (!GV->hasInitializer() && GV->use_size() == 0) {
      // No references to uninitialized global variable?
      //cerr << "Removing global var: " << GV->getName() << endl;
      delete Mod->getGlobalList().remove(GI);
      // Remove moves iterator to point to the next one automatically
      Changed = true;
    } else {
      ++GI;
    }
  }

  return Changed;
}

namespace {
  struct DeadCodeElimination : public FunctionPass {

    // Pass Interface...
    virtual bool doInitialization(Module *M) {
      return RemoveUnusedGlobalValues(M);
    }
    
    // It is possible that we may require multiple passes over the code to fully
    // eliminate dead code.  Iterate until we are done.
    //
    virtual bool runOnFunction(Function *F) {
      bool Changed = false;
      while (DoDCEPass(F)) Changed = true;
      return Changed;
    }
    
    virtual bool doFinalization(Module *M) {
      return RemoveUnusedGlobalValues(M);
    }
  };
}

Pass *createDeadCodeEliminationPass() {
  return new DeadCodeElimination();
}
