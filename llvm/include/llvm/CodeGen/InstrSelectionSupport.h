// $Id$ -*-c++-*-
//***************************************************************************
// File:
//	InstrSelectionSupport.h
// 
// Purpose:
//	Target-independent instruction selection code.
//      See SparcInstrSelection.cpp for usage.
//      
// History:
//	10/10/01	 -  Vikram Adve  -  Created
//**************************************************************************/

#ifndef LLVM_CODEGEN_INSTR_SELECTION_SUPPORT_H
#define LLVM_CODEGEN_INSTR_SELECTION_SUPPORT_H

#include "llvm/Instruction.h"
#include "llvm/CodeGen/MachineInstr.h"
class Method;
class InstrForest;
class MachineInstr;
class InstructionNode;
class TmpInstruction;
class ConstPoolVal;
class TargetMachine;

//************************ Exported Functions ******************************/


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


//------------------------------------------------------------------------ 
// Function Set2OperandsFromInstr
// Function Set3OperandsFromInstr
// 
// Purpose:
// 
// For the common case of 2- and 3-operand arithmetic/logical instructions,
// set the m/c instr. operands directly from the VM instruction's operands.
// Check whether the first or second operand is 0 and can use a dedicated
// "0" register.
// Check whether the second operand should use an immediate field or register.
// (First and third operands are never immediates for such instructions.)
// 
// Arguments:
// canDiscardResult: Specifies that the result operand can be discarded
//		     by using the dedicated "0"
// 
// op1position, op2position and resultPosition: Specify in which position
//		     in the machine instruction the 3 operands (arg1, arg2
//		     and result) should go.
// 
// RETURN VALUE: unsigned int flags, where
//	flags & 0x01	=> operand 1 is constant and needs a register
//	flags & 0x02	=> operand 2 is constant and needs a register
//------------------------------------------------------------------------ 

void		Set2OperandsFromInstr	(MachineInstr* minstr,
					 InstructionNode* vmInstrNode,
					 const TargetMachine& targetMachine,
					 bool canDiscardResult = false,
					 int op1Position = 0,
					 int resultPosition = 1);

void		Set3OperandsFromInstr	(MachineInstr* minstr,
					 InstructionNode* vmInstrNode,
					 const TargetMachine& targetMachine,
					 bool canDiscardResult = false,
					 int op1Position = 0,
					 int op2Position = 1,
					 int resultPosition = 2);


//---------------------------------------------------------------------------
// Function: ChooseRegOrImmed
// 
// Purpose:
// 
//---------------------------------------------------------------------------

MachineOperand::MachineOperandType
		ChooseRegOrImmed        (Value* val,
                                         MachineOpCode opCode,
                                         const TargetMachine& targetMachine,
                                         bool canUseImmed,
                                         unsigned int& getMachineRegNum,
                                         int64_t& getImmedValue);

//**************************************************************************/

#endif
