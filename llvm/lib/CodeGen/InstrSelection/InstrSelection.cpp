// $Id$ -*-c++-*-
//***************************************************************************
// File:
//	InstrSelection.cpp
// 
// Purpose:
//	Machine-independent driver file for instruction selection.
//	This file constructs a forest of BURG instruction trees and then
//      use the BURG-generated tree grammar (BURM) to find the optimal
//      instruction sequences for a given machine.
//	
// History:
//	7/02/01	 -  Vikram Adve  -  Created
//**************************************************************************/


#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Type.h"
#include "llvm/iMemory.h"
#include "llvm/Instruction.h"
#include "llvm/BasicBlock.h"
#include "llvm/Method.h"

static bool SelectInstructionsForTree(InstrTreeNode* treeRoot, int goalnt,
				      TargetMachine &Target);


enum SelectDebugLevel_t {
  Select_NoDebugInfo,
  Select_PrintMachineCode, 
  Select_DebugInstTrees, 
  Select_DebugBurgTrees,
};

// Enable Debug Options to be specified on the command line
cl::Enum<enum SelectDebugLevel_t> SelectDebugLevel("dselect", cl::NoFlags,
   "enable instruction selection debugging information",
   clEnumValN(Select_NoDebugInfo,      "n", "disable debug output"),
   clEnumValN(Select_PrintMachineCode, "y", "print generated machine code"),
   clEnumValN(Select_DebugInstTrees,   "i", "print debugging info for instruction selection "),
   clEnumValN(Select_DebugBurgTrees,   "b", "print burg trees"), 0);



//---------------------------------------------------------------------------
// Entry point for instruction selection using BURG.
// Returns true if instruction selection failed, false otherwise.
//---------------------------------------------------------------------------

bool
SelectInstructionsForMethod(Method* method, TargetMachine &Target)
{
  bool failed = false;
  
  //
  // Build the instruction trees to be given as inputs to BURG.
  // 
  InstrForest instrForest(method);
  
  if (SelectDebugLevel >= Select_DebugInstTrees)
    {
      cout << "\n\n*** Instruction trees for method "
	   << (method->hasName()? method->getName() : "")
	   << endl << endl;
      instrForest.dump();
    }
  
  //
  // Invoke BURG instruction selection for each tree
  // 
  const hash_set<InstructionNode*> &treeRoots = instrForest.getRootSet();
  for (hash_set<InstructionNode*>::const_iterator
	 treeRootIter = treeRoots.begin(); treeRootIter != treeRoots.end();
       ++treeRootIter)
    {
      InstrTreeNode* basicNode = *treeRootIter;
      
      // Invoke BURM to label each tree node with a state
      burm_label(basicNode);
      
      if (SelectDebugLevel >= Select_DebugBurgTrees)
	{
	  printcover(basicNode, 1, 0);
	  cerr << "\nCover cost == " << treecost(basicNode, 1, 0) << "\n\n";
	  printMatches(basicNode);
	}
      
      // Then recursively walk the tree to select instructions
      if (SelectInstructionsForTree(basicNode, /*goalnt*/1, Target))
	{
	  failed = true;
	  break;
	}
    }
  
  //
  // Record instructions in the vector for each basic block
  // 
  for (Method::iterator BI = method->begin(); BI != method->end(); ++BI)
    {
      MachineCodeForBasicBlock& bbMvec = (*BI)->getMachineInstrVec();
      for (BasicBlock::iterator II = (*BI)->begin(); II != (*BI)->end(); ++II)
	{
	  MachineCodeForVMInstr& mvec = (*II)->getMachineInstrVec();
	  for (unsigned i=0; i < mvec.size(); i++)
	    bbMvec.push_back(mvec[i]);
	}
    }
  
  if (SelectDebugLevel >= Select_PrintMachineCode)
    {
      cout << endl << "*** Machine instructions after INSTRUCTION SELECTION" << endl;
      PrintMachineInstructions(method);
    }
  
  return false;
}


//---------------------------------------------------------------------------
// Function: FoldGetElemChain
// 
// Purpose:
//   Fold a chain of GetElementPtr instructions into an equivalent
//   (Pointer, IndexVector) pair.  Returns the pointer Value, and
//   stores the resulting IndexVector in argument chainIdxVec.
//---------------------------------------------------------------------------

Value*
FoldGetElemChain(const InstructionNode* getElemInstrNode,
		 vector<ConstPoolVal*>& chainIdxVec)
{
  MemAccessInst* getElemInst = (MemAccessInst*)
    getElemInstrNode->getInstruction();
  
  // Initialize return values from the incoming instruction
  Value* ptrVal = getElemInst->getPtrOperand();
  chainIdxVec = getElemInst->getIndexVec(); // copies index vector values
  
  // Now chase the chain of getElementInstr instructions, if any
  InstrTreeNode* ptrChild = getElemInstrNode->leftChild();
  while (ptrChild->getOpLabel() == Instruction::GetElementPtr ||
	 ptrChild->getOpLabel() == GetElemPtrIdx)
    {
      // Child is a GetElemPtr instruction
      getElemInst = (MemAccessInst*)
	((InstructionNode*) ptrChild)->getInstruction();
      const vector<ConstPoolVal*>& idxVec = getElemInst->getIndexVec();
      
      // Get the pointer value out of ptrChild and *prepend* its index vector
      ptrVal = getElemInst->getPtrOperand();
      chainIdxVec.insert(chainIdxVec.begin(), idxVec.begin(), idxVec.end());
      
      ptrChild = ptrChild->leftChild();
    }
  
  return ptrVal;
}


//*********************** Private Functions *****************************/


//---------------------------------------------------------------------------
// Function SelectInstructionsForTree 
// 
// Recursively walk the tree to select instructions.
// Do this top-down so that child instructions can exploit decisions
// made at the child instructions.
// 
// E.g., if br(setle(reg,const)) decides the constant is 0 and uses
// a branch-on-integer-register instruction, then the setle node
// can use that information to avoid generating the SUBcc instruction.
//
// Note that this cannot be done bottom-up because setle must do this
// only if it is a child of the branch (otherwise, the result of setle
// may be used by multiple instructions).
//---------------------------------------------------------------------------

bool
SelectInstructionsForTree(InstrTreeNode* treeRoot, int goalnt,
			  TargetMachine &Target)
{
  // Use a static vector to avoid allocating a new one per VM instruction
  static MachineInstr* minstrVec[MAX_INSTR_PER_VMINSTR];
  
  // Get the rule that matches this node.
  // 
  int ruleForNode = burm_rule(treeRoot->state, goalnt);
  
  if (ruleForNode == 0)
    {
      cerr << "Could not match instruction tree for instr selection" << endl;
      assert(0);
      return true;
    }
  
  // Get this rule's non-terminals and the corresponding child nodes (if any)
  // 
  short *nts = burm_nts[ruleForNode];
  
  
  // First, select instructions for the current node and rule.
  // (If this is a list node, not an instruction, then skip this step).
  // This function is specific to the target architecture.
  // 
  if (treeRoot->opLabel != VRegListOp)
    {
      InstructionNode* instrNode = (InstructionNode*)treeRoot;
      assert(instrNode->getNodeType() == InstrTreeNode::NTInstructionNode);
    
      unsigned N = GetInstructionsByRule(instrNode, ruleForNode, nts, Target,
					 minstrVec);
      assert(N <= MAX_INSTR_PER_VMINSTR);
      for (unsigned i=0; i < N; i++)
	{
	  assert(minstrVec[i] != NULL);
	  instrNode->getInstruction()->addMachineInstruction(minstrVec[i]);
	}
    }
  
  // Then, recursively compile the child nodes, if any.
  // 
  if (nts[0])
    { // i.e., there is at least one kid
      InstrTreeNode* kids[2];
      int currentRule = ruleForNode;
      burm_kids(treeRoot, currentRule, kids);
    
      // First skip over any chain rules so that we don't visit
      // the current node again.
      // 
      while (ThisIsAChainRule(currentRule))
	{
	  currentRule = burm_rule(treeRoot->state, nts[0]);
	  nts = burm_nts[currentRule];
	  burm_kids(treeRoot, currentRule, kids);
	}
      
      // Now we have the first non-chain rule so we have found
      // the actual child nodes.  Recursively compile them.
      // 
      for (int i = 0; nts[i]; i++)
	{
	  assert(i < 2);
	  InstrTreeNode::InstrTreeNodeType nodeType = kids[i]->getNodeType();
	  if (nodeType == InstrTreeNode::NTVRegListNode ||
	      nodeType == InstrTreeNode::NTInstructionNode)
	    {
	      if (SelectInstructionsForTree(kids[i], nts[i], Target))
		return true;			// failure
	    }
	}
    }
  
  return false;				// success
}

