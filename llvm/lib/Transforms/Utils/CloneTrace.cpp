//===- CloneTrace.cpp - Clone a trace -------------------------------------===//
//
// This file implements the CloneTrace interface, which is used 
// when writing runtime optimizations. It takes a vector of basic blocks
// clones the basic blocks, removes internal phi nodes, adds it to the
// same function as the original (although there is no jump to it) and 
// returns the new vector of basic blocks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/iPHINode.h"
#include "llvm/Function.h"


//Clones the trace (a vector of basic blocks)
std::vector<BasicBlock *> CloneTrace(const std::vector<BasicBlock*> &origTrace) {

  std::vector<BasicBlock *> clonedTrace;
  std::map<const Value*, Value*> ValueMap;
  
  //First, loop over all the Basic Blocks in the trace and copy
  //them using CloneBasicBlock. Also fix the phi nodes during
  //this loop. To fix the phi nodes, we delete incoming branches
  //that are not in the trace.
  for(std::vector<BasicBlock *>::const_iterator T = origTrace.begin(),
	End = origTrace.end(); T != End; ++T) {

    //Clone Basic Block
    BasicBlock *clonedBlock = CloneBasicBlock(*T, ValueMap);
    
    //Add it to our new trace
    clonedTrace.push_back(clonedBlock);

    //Add this new mapping to our Value Map
    ValueMap[*T] = clonedBlock;

    //Add this cloned BB to the old BB's function
    (*T)->getParent()->getBasicBlockList().push_back(clonedBlock);

    //Loop over the phi instructions and delete operands
    //that are from blocks not in the trace
    //only do this if we are NOT the first block
    if(T != origTrace.begin()) {
      for (BasicBlock::iterator I = clonedBlock->begin();
	   PHINode *PN = dyn_cast<PHINode>(I); ++I) {
	//get incoming value for the previous BB
	Value *V = PN->getIncomingValueForBlock(*(T-1));
	assert(V && "No incoming value from a BasicBlock in our trace!");
	
	//remap our phi node to point to incoming value
	ValueMap[*&I] = V;
	
	//remove phi node
	clonedBlock->getInstList().erase(PN);
      }
    }
  }

  //Second loop to do the remapping
  for(std::vector<BasicBlock *>::const_iterator BB = clonedTrace.begin(),
	BE = clonedTrace.end(); BB != BE; ++BB) {
    for(BasicBlock::iterator I = (*BB)->begin(); I != (*BB)->end(); ++I) {
      
      //Loop over all the operands of the instruction
      for(unsigned op=0, E = I->getNumOperands(); op != E; ++op) {
	const Value *Op = I->getOperand(op);
	
	//Get it out of the value map
	Value *V = ValueMap[Op];

	//If not in the value map, then its outside our trace so ignore
	if(V != 0)
	  I->setOperand(op,V);
      }
    }
  }
  
  //return new vector of basic blocks
  return clonedTrace;
}
