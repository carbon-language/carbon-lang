// $Id$ -*-c++-*-
//***************************************************************************
// File:
//	InstrSelection.h
// 
// Purpose:
//	External interface to instruction selection.
// 
// History:
//	7/02/01	 -  Vikram Adve  -  Created
//**************************************************************************/

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


/************************* Required Functions *******************************
 * Target-dependent functions that MUST be implemented for each target.
 ***************************************************************************/

const unsigned MAX_INSTR_PER_VMINSTR = 8;

const Instruction::OtherOps TMP_INSTRUCTION_OPCODE = Instruction::UserOp1;

extern unsigned	GetInstructionsByRule	(InstructionNode* subtreeRoot,
					 int ruleForNode,
					 short* nts,
					 TargetMachine &Target,
					 MachineInstr** minstrVec);

extern unsigned	GetInstructionsForProlog(BasicBlock* entryBB,
					 TargetMachine &Target,
					 MachineInstr** minstrVec);

extern unsigned	GetInstructionsForEpilog(BasicBlock* anExitBB,
					 TargetMachine &Target,
					 MachineInstr** minstrVec);

extern bool	ThisIsAChainRule	(int eruleno);


//************************ Exported Functions ******************************/


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


//************************ Exported Data Types *****************************/


//---------------------------------------------------------------------------
// class TmpInstruction
//
//   This class represents temporary intermediate values
//   used within the machine code for a VM instruction
//---------------------------------------------------------------------------

class TmpInstruction : public Instruction {
  TmpInstruction (const TmpInstruction  &ci)
    : Instruction(ci.getType(), ci.getOpcode())
  {
    Operands.reserve(2);
    Operands.push_back(Use(Operands[0], this));
    Operands.push_back(Use(Operands[1], this));
  }
public:
  // Constructor that uses the type of S1 as the type of the temporary.
  // s1 must be a valid value.  s2 may be NULL.
  TmpInstruction(OtherOps opcode, Value *s1, Value* s2, const string &name="")
    : Instruction(s1->getType(), opcode, name)
  {
    assert(s1 != NULL && "Use different constructor if both operands are 0");
    Initialize(opcode, s1, s2);
  }
  
  // Constructor that allows the type of the temporary to be specified.
  // Both S1 and S2 may be NULL.
  TmpInstruction(OtherOps opcode, const Type* tmpType,
                 Value *s1, Value* s2, const string &name = "")
    : Instruction(tmpType, opcode, name)
  {
    Initialize(opcode, s1, s2);
  }
  
  virtual Instruction *clone() const { return new TmpInstruction(*this); }
  virtual const char *getOpcodeName() const {
    return "userOp1";
  }
  
private:
  void Initialize(OtherOps opcode, Value *s1, Value* s2) {
    assert(opcode==TMP_INSTRUCTION_OPCODE && "Tmp instruction opcode invalid");
    Operands.reserve(s1 && s2? 2 : ((s1 || s2)? 1 : 0));
    if (s1)
      Operands.push_back(Use(s1, this));
    if (s2)
      Operands.push_back(Use(s2, this));
  }
};

//**************************************************************************/

#endif
