// $Id$ -*-c++-*-
//***************************************************************************
// File:
//	InstrSelection.cpp
// 
// Purpose:
//	Machine-independent driver file for instruction selection.
//	This file constructs a forest of BURG instruction trees and then
//      uses the BURG-generated tree grammar (BURM) to find the optimal
//      instruction sequences for a given machine.
//	
// History:
//	7/02/01	 -  Vikram Adve  -  Created
//**************************************************************************/


#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/InstrSelectionSupport.h"
#include "llvm/CodeGen/InstrForest.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/CodeGen/MachineCodeForMethod.h"
#include "llvm/Target/MachineRegInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/BasicBlock.h"
#include "llvm/Function.h"
#include "llvm/iPHINode.h"
#include "Support/CommandLine.h"
#include <iostream>
using std::cerr;

//******************** Internal Data Declarations ************************/


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


//******************** Forward Function Declarations ***********************/


static bool SelectInstructionsForTree   (InstrTreeNode* treeRoot,
                                         int goalnt,
                                         TargetMachine &target);

static void PostprocessMachineCodeForTree(InstructionNode* instrNode,
                                          int ruleForNode,
                                          short* nts,
                                          TargetMachine &target);

static void InsertCode4AllPhisInMeth(Function *F, TargetMachine &target);



//******************* Externally Visible Functions *************************/


//---------------------------------------------------------------------------
// Entry point for instruction selection using BURG.
// Returns true if instruction selection failed, false otherwise.
//---------------------------------------------------------------------------

bool
SelectInstructionsForMethod(Function *F, TargetMachine &target)
{
  bool failed = false;
  
  //
  // Build the instruction trees to be given as inputs to BURG.
  // 
  InstrForest instrForest(F);
  
  if (SelectDebugLevel >= Select_DebugInstTrees)
    {
      cerr << "\n\n*** Input to instruction selection for function "
	   << F->getName() << "\n\n";
      F->dump();
      
      cerr << "\n\n*** Instruction trees for function "
	   << F->getName() << "\n\n";
      instrForest.dump();
    }
  
  //
  // Invoke BURG instruction selection for each tree
  // 
  for (InstrForest::const_root_iterator RI = instrForest.roots_begin();
       RI != instrForest.roots_end(); ++RI)
    {
      InstructionNode* basicNode = *RI;
      assert(basicNode->parent() == NULL && "A `root' node has a parent?"); 
      
      // Invoke BURM to label each tree node with a state
      burm_label(basicNode);
      
      if (SelectDebugLevel >= Select_DebugBurgTrees)
	{
	  printcover(basicNode, 1, 0);
	  cerr << "\nCover cost == " << treecost(basicNode, 1, 0) << "\n\n";
	  printMatches(basicNode);
	}
      
      // Then recursively walk the tree to select instructions
      if (SelectInstructionsForTree(basicNode, /*goalnt*/1, target))
	{
	  failed = true;
	  break;
	}
    }
  
  //
  // Record instructions in the vector for each basic block
  // 
  for (Function::iterator BI = F->begin(), BE = F->end(); BI != BE; ++BI)
    {
      MachineCodeForBasicBlock& bbMvec = (*BI)->getMachineInstrVec();
      for (BasicBlock::iterator II = (*BI)->begin(); II != (*BI)->end(); ++II)
	{
	  MachineCodeForInstruction &mvec =MachineCodeForInstruction::get(*II);
	  for (unsigned i=0; i < mvec.size(); i++)
	    bbMvec.push_back(mvec[i]);
	}
    }

  // Insert phi elimination code -- added by Ruchira
  InsertCode4AllPhisInMeth(F, target);

  
  if (SelectDebugLevel >= Select_PrintMachineCode)
    {
      cerr << "\n*** Machine instructions after INSTRUCTION SELECTION\n";
      MachineCodeForMethod::get(F).dump();
    }
  
  return false;
}


//*********************** Private Functions *****************************/


//-------------------------------------------------------------------------
// Thid method inserts a copy instruction to a predecessor BB as a result
// of phi elimination.
//-------------------------------------------------------------------------

void
InsertPhiElimInstructions(BasicBlock *BB, const vector<MachineInstr*>& CpVec)
{ 
  Instruction *TermInst = (Instruction*)BB->getTerminator();
  MachineCodeForInstruction &MC4Term =MachineCodeForInstruction::get(TermInst);
  MachineInstr *FirstMIOfTerm = *( MC4Term.begin() );
  
  assert( FirstMIOfTerm && "No Machine Instrs for terminator" );
  
  // get an iterator to machine instructions in the BB
  MachineCodeForBasicBlock& bbMvec = BB->getMachineInstrVec();
  MachineCodeForBasicBlock::iterator MCIt =  bbMvec.begin();
  
  // find the position of first machine instruction generated by the
  // terminator of this BB
  for( ; (MCIt != bbMvec.end()) && (*MCIt != FirstMIOfTerm) ; ++MCIt )
    ;
  assert( MCIt != bbMvec.end() && "Start inst of terminator not found");
  
  // insert the copy instructions just before the first machine instruction
  // generated for the terminator
  bbMvec.insert(MCIt, CpVec.begin(), CpVec.end());
  
  //cerr << "\nPhiElimination copy inst: " <<   *CopyInstVec[0];
}


//-------------------------------------------------------------------------
// This method inserts phi elimination code for all BBs in a method
//-------------------------------------------------------------------------

void
InsertCode4AllPhisInMeth(Function *F, TargetMachine &target)
{
  // for all basic blocks in function
  //
  for (Function::iterator BI = F->begin(); BI != F->end(); ++BI) {

    BasicBlock *BB = *BI;
    const BasicBlock::InstListType &InstList = BB->getInstList();
    BasicBlock::InstListType::const_iterator  IIt = InstList.begin();

    // for all instructions in the basic block
    //
    for( ; IIt != InstList.end(); ++IIt ) {

      if (PHINode *PN = dyn_cast<PHINode>(*IIt)) {
        // FIXME: This is probably wrong...
	Value *PhiCpRes = new PHINode(PN->getType(), "PhiCp:");
        
	// for each incoming value of the phi, insert phi elimination
	//
        for (unsigned i = 0; i < PN->getNumIncomingValues(); ++i)
          { // insert the copy instruction to the predecessor BB
            vector<MachineInstr*> mvec, CpVec;
            target.getRegInfo().cpValue2Value(PN->getIncomingValue(i), PhiCpRes,
                                              mvec);
            for (vector<MachineInstr*>::iterator MI=mvec.begin();
                 MI != mvec.end(); ++MI)
              {
                vector<MachineInstr*> CpVec2 =
                  FixConstantOperandsForInstr(PN, *MI, target);
                CpVec2.push_back(*MI);
                CpVec.insert(CpVec.end(), CpVec2.begin(), CpVec2.end());
              }
            
            InsertPhiElimInstructions(PN->getIncomingBlock(i), CpVec);
          }
        
        vector<MachineInstr*> mvec;
        target.getRegInfo().cpValue2Value(PhiCpRes, PN, mvec);
        
	// get an iterator to machine instructions in the BB
	MachineCodeForBasicBlock& bbMvec = BB->getMachineInstrVec();

	bbMvec.insert( bbMvec.begin(), mvec.begin(), mvec.end());
      }
      else break;   // since PHI nodes can only be at the top
      
    }  // for each Phi Instr in BB
  } // for all BBs in function
}


//---------------------------------------------------------------------------
// Function PostprocessMachineCodeForTree
// 
// Apply any final cleanups to machine code for the root of a subtree
// after selection for all its children has been completed.
//---------------------------------------------------------------------------

static void
PostprocessMachineCodeForTree(InstructionNode* instrNode,
                              int ruleForNode,
                              short* nts,
                              TargetMachine &target)
{
  // Fix up any constant operands in the machine instructions to either
  // use an immediate field or to load the constant into a register
  // Walk backwards and use direct indexes to allow insertion before current
  // 
  Instruction* vmInstr = instrNode->getInstruction();
  MachineCodeForInstruction &mvec = MachineCodeForInstruction::get(vmInstr);
  for (int i = (int) mvec.size()-1; i >= 0; i--)
    {
      std::vector<MachineInstr*> loadConstVec =
        FixConstantOperandsForInstr(vmInstr, mvec[i], target);
      
      if (loadConstVec.size() > 0)
        mvec.insert(mvec.begin()+i, loadConstVec.begin(), loadConstVec.end());
    }
}

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
			  TargetMachine &target)
{
  // Get the rule that matches this node.
  // 
  int ruleForNode = burm_rule(treeRoot->state, goalnt);
  
  if (ruleForNode == 0)
    {
      cerr << "Could not match instruction tree for instr selection\n";
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
      vector<MachineInstr*> minstrVec;
      
      InstructionNode* instrNode = (InstructionNode*)treeRoot;
      assert(instrNode->getNodeType() == InstrTreeNode::NTInstructionNode);
      
      GetInstructionsByRule(instrNode, ruleForNode, nts, target, minstrVec);
      
      MachineCodeForInstruction &mvec = 
        MachineCodeForInstruction::get(instrNode->getInstruction());
      mvec.insert(mvec.end(), minstrVec.begin(), minstrVec.end());
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
	      if (SelectInstructionsForTree(kids[i], nts[i], target))
		return true;			// failure
	    }
	}
    }
  
  // Finally, do any postprocessing on this node after its children
  // have been translated
  // 
  if (treeRoot->opLabel != VRegListOp)
    {
      InstructionNode* instrNode = (InstructionNode*)treeRoot;
      PostprocessMachineCodeForTree(instrNode, ruleForNode, nts, target);
    }
  
  return false;				// success
}

