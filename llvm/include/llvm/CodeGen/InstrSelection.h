// $Id$ -*-c++-*-
//***************************************************************************
// File:
//	InstrSelection.h
// 
// Purpose:
//	
// History:
//	7/02/01	 -  Vikram Adve  -  Created
//***************************************************************************

#ifndef LLVM_CODEGEN_INSTR_SELECTION_H
#define LLVM_CODEGEN_INSTR_SELECTION_H

#include "llvm/Instruction.h"
class Method;
class InstrForest;
class MachineInstr;
class InstructionNode;
class TmpInstruction;
class ConstPoolVal;
class TargetMachine;

//---------------------------------------------------------------------------
// GLOBAL data and an external function that must be implemented
// for each architecture.
//---------------------------------------------------------------------------

const unsigned MAX_INSTR_PER_VMINSTR = 8;

extern unsigned	GetInstructionsByRule	(InstructionNode* subtreeRoot,
					 int ruleForNode,
					 short* nts,
					 TargetMachine &Target,
					 MachineInstr** minstrVec);

extern bool	ThisIsAChainRule	(int eruleno);


//************************ Exported Data Types *****************************/


//---------------------------------------------------------------------------
// Function: SelectInstructionsForMethod
// 
// Purpose:
//   Entry point for instruction selection using BURG.
//   Returns true if instruction selection failed, false otherwise.
//   Implemented in machine-specific instruction selection file.
//---------------------------------------------------------------------------

bool		SelectInstructionsForMethod	(Method* method,
						 TargetMachine &Target);

//---------------------------------------------------------------------------
// Function: FoldGetElemChain
// 
// Purpose:
//   Fold a chain of GetElementPtr instructions into an equivalent
//   (Pointer, IndexVector) pair.  Returns the pointer Value, and
//   stores the resulting IndexVector in argument chainIdxVec.
//---------------------------------------------------------------------------

Value*		FoldGetElemChain    (const InstructionNode* getElemInstrNode,
				     vector<ConstPoolVal*>& chainIdxVec);


//---------------------------------------------------------------------------
// class TmpInstruction
//
//   This class represents temporary intermediate values
//   used within the machine code for a VM instruction
//---------------------------------------------------------------------------

class TmpInstruction : public Instruction {
  TmpInstruction (const TmpInstruction  &CI) : Instruction(CI.getType(), CI.getOpcode()) {
    Operands.reserve(2);
    Operands.push_back(Use(Operands[0], this));
    Operands.push_back(Use(Operands[1], this));
  }
public:
  TmpInstruction(OtherOps Opcode, Value *S1, Value* S2, const string &Name = "")
    : Instruction(S1->getType(), Opcode, Name)
  {
    assert(Opcode == UserOp1 && "Tmp instruction opcode invalid!");
    Operands.reserve(S2? 2 : 1);
    Operands.push_back(Use(S1, this));
    if (S2)
      Operands.push_back(Use(S2, this));
  }
  
  virtual Instruction *clone() const { return new TmpInstruction(*this); }
  virtual const char *getOpcodeName() const {
    return "userOp1";
  }
};

//**************************************************************************/

#endif
