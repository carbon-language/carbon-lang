//===- InstrSelection.cpp - Machine Independent Inst Selection Driver -----===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Machine-independent driver file for instruction selection.  This file
// constructs a forest of BURG instruction trees and then uses the
// BURG-generated tree grammar (BURM) to find the optimal instruction sequences
// for a given machine.
//	
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/Function.h"
#include "llvm/IntrinsicLowering.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/InstrForest.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetMachine.h"
#include "../SparcV9RegInfo.h"
#include "Support/CommandLine.h"
#include "Support/LeakDetector.h"

namespace llvm {
  std::vector<MachineInstr*>
  FixConstantOperandsForInstr(Instruction *I, MachineInstr *MI,
                              TargetMachine &TM);
}

namespace {
  //===--------------------------------------------------------------------===//
  // SelectDebugLevel - Allow command line control over debugging.
  //
  enum SelectDebugLevel_t {
    Select_NoDebugInfo,
    Select_PrintMachineCode, 
    Select_DebugInstTrees, 
    Select_DebugBurgTrees,
  };
  
  // Enable Debug Options to be specified on the command line
  cl::opt<SelectDebugLevel_t>
  SelectDebugLevel("dselect", cl::Hidden,
                   cl::desc("enable instruction selection debug information"),
                   cl::values(
     clEnumValN(Select_NoDebugInfo,      "n", "disable debug output"),
     clEnumValN(Select_PrintMachineCode, "y", "print generated machine code"),
     clEnumValN(Select_DebugInstTrees,   "i",
                "print debugging info for instruction selection"),
     clEnumValN(Select_DebugBurgTrees,   "b", "print burg trees"),
                              0));


  //===--------------------------------------------------------------------===//
  //  InstructionSelection Pass
  //
  // This is the actual pass object that drives the instruction selection
  // process.
  //
  class InstructionSelection : public FunctionPass {
    TargetMachine &Target;
    void InsertCodeForPhis(Function &F);
    void InsertPhiElimInstructions(BasicBlock *BB,
                                   const std::vector<MachineInstr*>& CpVec);
    void SelectInstructionsForTree(InstrTreeNode* treeRoot, int goalnt);
    void PostprocessMachineCodeForTree(InstructionNode* instrNode,
                                       int ruleForNode, short* nts);
  public:
    InstructionSelection(TargetMachine &TM) : Target(TM) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }
    
    bool runOnFunction(Function &F);
    virtual const char *getPassName() const { return "Instruction Selection"; }
  };
}


TmpInstruction::TmpInstruction(MachineCodeForInstruction& mcfi,
                               Value *s1, Value *s2, const std::string &name)
  : Instruction(s1->getType(), Instruction::UserOp1, name)
{
  mcfi.addTemp(this);

  Operands.push_back(Use(s1, this));  // s1 must be non-null
  if (s2)
    Operands.push_back(Use(s2, this));

  // TmpInstructions should not be garbage checked.
  LeakDetector::removeGarbageObject(this);
}
  
// Constructor that requires the type of the temporary to be specified.
// Both S1 and S2 may be NULL.(
TmpInstruction::TmpInstruction(MachineCodeForInstruction& mcfi,
                               const Type *Ty, Value *s1, Value* s2,
                               const std::string &name)
  : Instruction(Ty, Instruction::UserOp1, name)
{
  mcfi.addTemp(this);

  if (s1) 
    Operands.push_back(Use(s1, this));
  if (s2)
    Operands.push_back(Use(s2, this));

  // TmpInstructions should not be garbage checked.
  LeakDetector::removeGarbageObject(this);
}

bool InstructionSelection::runOnFunction(Function &F) {
  // First pass - Walk the function, lowering any calls to intrinsic functions
  // which the instruction selector cannot handle.
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; )
      if (CallInst *CI = dyn_cast<CallInst>(I++))
        if (Function *F = CI->getCalledFunction())
          switch (F->getIntrinsicID()) {
          case Intrinsic::not_intrinsic:
          case Intrinsic::vastart:
          case Intrinsic::vacopy:
          case Intrinsic::vaend:
            // We directly implement these intrinsics.  Note that this knowledge
            // is incestuously entangled with the code in
            // SparcInstrSelection.cpp and must be updated when it is updated.
            // Since ALL of the code in this library is incestuously intertwined
            // with it already and sparc specific, we will live with this.
            break;
          default:
            // All other intrinsic calls we must lower.
            Instruction *Before = CI->getPrev();
            Target.getIntrinsicLowering().LowerIntrinsicCall(CI);
            if (Before) {        // Move iterator to instruction after call
              I = Before;  ++I;
            } else {
              I = BB->begin();
            }
          }

  //
  // Build the instruction trees to be given as inputs to BURG.
  // 
  InstrForest instrForest(&F);
  
  if (SelectDebugLevel >= Select_DebugInstTrees) {
    std::cerr << "\n\n*** Input to instruction selection for function "
              << F.getName() << "\n\n" << F
              << "\n\n*** Instruction trees for function "
              << F.getName() << "\n\n";
    instrForest.dump();
  }
  
  //
  // Invoke BURG instruction selection for each tree
  // 
  for (InstrForest::const_root_iterator RI = instrForest.roots_begin();
       RI != instrForest.roots_end(); ++RI) {
    InstructionNode* basicNode = *RI;
    assert(basicNode->parent() == NULL && "A `root' node has a parent?"); 
      
    // Invoke BURM to label each tree node with a state
    burm_label(basicNode);
      
    if (SelectDebugLevel >= Select_DebugBurgTrees) {
      printcover(basicNode, 1, 0);
      std::cerr << "\nCover cost == " << treecost(basicNode, 1, 0) <<"\n\n";
      printMatches(basicNode);
    }
      
    // Then recursively walk the tree to select instructions
    SelectInstructionsForTree(basicNode, /*goalnt*/1);
  }
  
  //
  // Create the MachineBasicBlock records and add all of the MachineInstrs
  // defined in the MachineCodeForInstruction objects to also live in the
  // MachineBasicBlock objects.
  // 
  MachineFunction &MF = MachineFunction::get(&F);
  for (Function::iterator BI = F.begin(), BE = F.end(); BI != BE; ++BI) {
    MachineBasicBlock *MCBB = new MachineBasicBlock(BI);
    MF.getBasicBlockList().push_back(MCBB);

    for (BasicBlock::iterator II = BI->begin(); II != BI->end(); ++II) {
      MachineCodeForInstruction &mvec = MachineCodeForInstruction::get(II);
      MCBB->insert(MCBB->end(), mvec.begin(), mvec.end());
    }
  }

  // Insert phi elimination code
  InsertCodeForPhis(F);
  
  if (SelectDebugLevel >= Select_PrintMachineCode) {
    std::cerr << "\n*** Machine instructions after INSTRUCTION SELECTION\n";
    MachineFunction::get(&F).dump();
  }
  
  return true;
}


//-------------------------------------------------------------------------
// This method inserts phi elimination code for all BBs in a method
//-------------------------------------------------------------------------

void
InstructionSelection::InsertCodeForPhis(Function &F) {
  // for all basic blocks in function
  //
  MachineFunction &MF = MachineFunction::get(&F);
  for (MachineFunction::iterator BB = MF.begin(); BB != MF.end(); ++BB) {
    for (BasicBlock::const_iterator IIt = BB->getBasicBlock()->begin();
         const PHINode *PN = dyn_cast<PHINode>(IIt); ++IIt) {
      // FIXME: This is probably wrong...
      Value *PhiCpRes = new PHINode(PN->getType(), "PhiCp:");

      // The leak detector shouldn't track these nodes.  They are not garbage,
      // even though their parent field is never filled in.
      //
      LeakDetector::removeGarbageObject(PhiCpRes);

      // for each incoming value of the phi, insert phi elimination
      //
      for (unsigned i = 0; i < PN->getNumIncomingValues(); ++i) {
        // insert the copy instruction to the predecessor BB
        std::vector<MachineInstr*> mvec, CpVec;
        Target.getRegInfo().cpValue2Value(PN->getIncomingValue(i), PhiCpRes,
                                          mvec);
        for (std::vector<MachineInstr*>::iterator MI=mvec.begin();
             MI != mvec.end(); ++MI) {
          std::vector<MachineInstr*> CpVec2 =
            FixConstantOperandsForInstr(const_cast<PHINode*>(PN), *MI, Target);
          CpVec2.push_back(*MI);
          CpVec.insert(CpVec.end(), CpVec2.begin(), CpVec2.end());
        }
        
        InsertPhiElimInstructions(PN->getIncomingBlock(i), CpVec);
      }
      
      std::vector<MachineInstr*> mvec;
      Target.getRegInfo().cpValue2Value(PhiCpRes, const_cast<PHINode*>(PN),
                                        mvec);
      BB->insert(BB->begin(), mvec.begin(), mvec.end());
    }  // for each Phi Instr in BB
  } // for all BBs in function
}

//-------------------------------------------------------------------------
// Thid method inserts a copy instruction to a predecessor BB as a result
// of phi elimination.
//-------------------------------------------------------------------------

void
InstructionSelection::InsertPhiElimInstructions(BasicBlock *BB,
                                        const std::vector<MachineInstr*>& CpVec)
{ 
  Instruction *TermInst = (Instruction*)BB->getTerminator();
  MachineCodeForInstruction &MC4Term = MachineCodeForInstruction::get(TermInst);
  MachineInstr *FirstMIOfTerm = MC4Term.front();
  assert (FirstMIOfTerm && "No Machine Instrs for terminator");

  MachineFunction &MF = MachineFunction::get(BB->getParent());

  // FIXME: if PHI instructions existed in the machine code, this would be
  // unnecessary.
  MachineBasicBlock *MBB = 0;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    if (I->getBasicBlock() == BB) {
      MBB = I;
      break;
    }

  MachineBasicBlock::iterator MCIt = FirstMIOfTerm;

  assert(MCIt != MBB->end() && "Start inst of terminator not found");
  
  // insert the copy instructions just before the first machine instruction
  // generated for the terminator
  MBB->insert(MCIt, CpVec.begin(), CpVec.end());
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

void 
InstructionSelection::SelectInstructionsForTree(InstrTreeNode* treeRoot,
                                                int goalnt)
{
  // Get the rule that matches this node.
  // 
  int ruleForNode = burm_rule(treeRoot->state, goalnt);
  
  if (ruleForNode == 0) {
    std::cerr << "Could not match instruction tree for instr selection\n";
    abort();
  }
  
  // Get this rule's non-terminals and the corresponding child nodes (if any)
  // 
  short *nts = burm_nts[ruleForNode];
  
  // First, select instructions for the current node and rule.
  // (If this is a list node, not an instruction, then skip this step).
  // This function is specific to the target architecture.
  // 
  if (treeRoot->opLabel != VRegListOp) {
    std::vector<MachineInstr*> minstrVec;
      
    InstructionNode* instrNode = (InstructionNode*)treeRoot;
    assert(instrNode->getNodeType() == InstrTreeNode::NTInstructionNode);
      
    GetInstructionsByRule(instrNode, ruleForNode, nts, Target, minstrVec);
      
    MachineCodeForInstruction &mvec = 
      MachineCodeForInstruction::get(instrNode->getInstruction());
    mvec.insert(mvec.end(), minstrVec.begin(), minstrVec.end());
  }
  
  // Then, recursively compile the child nodes, if any.
  // 
  if (nts[0]) {
    // i.e., there is at least one kid
    InstrTreeNode* kids[2];
    int currentRule = ruleForNode;
    burm_kids(treeRoot, currentRule, kids);
    
    // First skip over any chain rules so that we don't visit
    // the current node again.
    // 
    while (ThisIsAChainRule(currentRule)) {
      currentRule = burm_rule(treeRoot->state, nts[0]);
      nts = burm_nts[currentRule];
      burm_kids(treeRoot, currentRule, kids);
    }
      
    // Now we have the first non-chain rule so we have found
    // the actual child nodes.  Recursively compile them.
    // 
    for (unsigned i = 0; nts[i]; i++) {
      assert(i < 2);
      InstrTreeNode::InstrTreeNodeType nodeType = kids[i]->getNodeType();
      if (nodeType == InstrTreeNode::NTVRegListNode ||
          nodeType == InstrTreeNode::NTInstructionNode)
        SelectInstructionsForTree(kids[i], nts[i]);
    }
  }
  
  // Finally, do any post-processing on this node after its children
  // have been translated
  // 
  if (treeRoot->opLabel != VRegListOp)
    PostprocessMachineCodeForTree((InstructionNode*)treeRoot, ruleForNode, nts);
}

//---------------------------------------------------------------------------
// Function PostprocessMachineCodeForTree
// 
// Apply any final cleanups to machine code for the root of a subtree
// after selection for all its children has been completed.
//
void
InstructionSelection::PostprocessMachineCodeForTree(InstructionNode* instrNode,
                                                    int ruleForNode,
                                                    short* nts) 
{
  // Fix up any constant operands in the machine instructions to either
  // use an immediate field or to load the constant into a register
  // Walk backwards and use direct indexes to allow insertion before current
  // 
  Instruction* vmInstr = instrNode->getInstruction();
  MachineCodeForInstruction &mvec = MachineCodeForInstruction::get(vmInstr);
  for (unsigned i = mvec.size(); i != 0; --i) {
    std::vector<MachineInstr*> loadConstVec =
      FixConstantOperandsForInstr(vmInstr, mvec[i-1], Target);
      
    mvec.insert(mvec.begin()+i-1, loadConstVec.begin(), loadConstVec.end());
  }
}


//===----------------------------------------------------------------------===//
// createInstructionSelectionPass - Public entrypoint for instruction selection
// and this file as a whole...
//
FunctionPass *llvm::createInstructionSelectionPass(TargetMachine &TM) {
  return new InstructionSelection(TM);
}
