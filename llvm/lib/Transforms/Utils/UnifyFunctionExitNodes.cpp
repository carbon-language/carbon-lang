//===- UnifyFunctionExitNodes.cpp - Make all functions have a single exit -===//
//
// This file provides several routines that are useful for simplifying CFGs in
// various ways...
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/UnifyMethodExitNodes.h"
#include "llvm/BasicBlock.h"
#include "llvm/Function.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/Type.h"
using std::vector;

AnalysisID UnifyMethodExitNodes::ID(AnalysisID::create<UnifyMethodExitNodes>());


// UnifyAllExitNodes - Unify all exit nodes of the CFG by creating a new
// BasicBlock, and converting all returns to unconditional branches to this
// new basic block.  The singular exit node is returned.
//
// If there are no return stmts in the Function, a null pointer is returned.
//
bool UnifyMethodExitNodes::doit(Function *M, BasicBlock *&ExitNode) {
  // Loop over all of the blocks in a function, tracking all of the blocks that
  // return.
  //
  vector<BasicBlock*> ReturningBlocks;
  for(Function::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (isa<ReturnInst>((*I)->getTerminator()))
      ReturningBlocks.push_back(*I);

  if (ReturningBlocks.empty()) {
    ExitNode = 0;
    return false;                      // No blocks return
  } else if (ReturningBlocks.size() == 1) {
    ExitNode = ReturningBlocks.front();    // Already has a single return block
    return false;
  }

  // Otherwise, we need to insert a new basic block into the function, add a PHI
  // node (if the function returns a value), and convert all of the return 
  // instructions into unconditional branches.
  //
  BasicBlock *NewRetBlock = new BasicBlock("UnifiedExitNode", M);

  if (M->getReturnType() != Type::VoidTy) {
    // If the function doesn't return void... add a PHI node to the block...
    PHINode *PN = new PHINode(M->getReturnType());
    NewRetBlock->getInstList().push_back(PN);

    // Add an incoming element to the PHI node for every return instruction that
    // is merging into this new block...
    for (vector<BasicBlock*>::iterator I = ReturningBlocks.begin(), 
                                       E = ReturningBlocks.end(); I != E; ++I)
      PN->addIncoming((*I)->getTerminator()->getOperand(0), *I);

    // Add a return instruction to return the result of the PHI node...
    NewRetBlock->getInstList().push_back(new ReturnInst(PN));
  } else {
    // If it returns void, just add a return void instruction to the block
    NewRetBlock->getInstList().push_back(new ReturnInst());
  }

  // Loop over all of the blocks, replacing the return instruction with an
  // unconditional branch.
  //
  for (vector<BasicBlock*>::iterator I = ReturningBlocks.begin(), 
                                     E = ReturningBlocks.end(); I != E; ++I) {
    delete (*I)->getInstList().pop_back();  // Remove the return insn
    (*I)->getInstList().push_back(new BranchInst(NewRetBlock));
  }
  ExitNode = NewRetBlock;
  return true;
}
