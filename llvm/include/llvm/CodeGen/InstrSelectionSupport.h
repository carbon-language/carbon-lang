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
class InstructionNode;
class TargetMachine;


//---------------------------------------------------------------------------
// Function GetConstantValueAsSignedInt
// 
// Convenience function to get the value of an integer constant, for an
// appropriate integer or non-integer type that can be held in an integer.
// The type of the argument must be the following:
//      Signed or unsigned integer
//      Boolean
//      Pointer
// 
// isValidConstant is set to true if a valid constant was found.
//---------------------------------------------------------------------------

int64_t         GetConstantValueAsSignedInt     (const Value *V,
                                                 bool &isValidConstant);


//---------------------------------------------------------------------------
// Function: FoldGetElemChain
// 
// Purpose:
//   Fold a chain of GetElementPtr instructions into an equivalent
//   (Pointer, IndexVector) pair.  Returns the pointer Value, and
//   stores the resulting IndexVector in argument chainIdxVec.
//---------------------------------------------------------------------------

Value*		FoldGetElemChain    (const InstructionNode* getElemInstrNode,
				     std::vector<Value*>& chainIdxVec);


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


//------------------------------------------------------------------------ 
// Common machine instruction operand combinations
// to simplify code generation.
//------------------------------------------------------------------------ 

inline MachineInstr*
Create1OperandInstr(MachineOpCode opCode, Value* argVal1)
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister, argVal1);
  return M;
}

inline MachineInstr*
Create2OperandInstr(MachineOpCode opCode, Value* argVal1, Value* argVal2)
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister, argVal1);
  M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister, argVal2);
  return M;
}

inline MachineInstr*
Create2OperandInstr(MachineOpCode opCode,
                    Value* argVal1, MachineOperand::MachineOperandType type1,
                    Value* argVal2, MachineOperand::MachineOperandType type2)
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandVal(0, type1, argVal1);
  M->SetMachineOperandVal(1, type2, argVal2);
  return M;
}


inline MachineInstr*
Create2OperandInstr_UImmed(MachineOpCode opCode,
                           unsigned int unextendedImmed, Value* argVal2)
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandConst(0, MachineOperand::MO_UnextendedImmed,
                               unextendedImmed);
  M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister, argVal2);
  return M;
}

inline MachineInstr*
Create2OperandInstr_SImmed(MachineOpCode opCode,
                           int signExtendedImmed, Value* argVal2)
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandConst(0, MachineOperand::MO_SignExtendedImmed,
                               signExtendedImmed);
  M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister, argVal2);
  return M;
}

inline MachineInstr*
Create2OperandInstr_Addr(MachineOpCode opCode,
                         Value* label, Value* argVal2)
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandVal(0, MachineOperand::MO_PCRelativeDisp,  label);
  M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister, argVal2);
  return M;
}

inline MachineInstr*
Create2OperandInstr_Reg(MachineOpCode opCode,
                        Value* argVal1, unsigned int regNum)
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister, argVal1);
  M->SetMachineOperandReg(1, regNum);
  return M;
}

inline MachineInstr*
Create2OperandInstr_Reg(MachineOpCode opCode,
                        unsigned int regNum1, unsigned int regNum2)
                 
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandReg(0, regNum1);
  M->SetMachineOperandReg(1, regNum2);
  return M;
}

inline MachineInstr*
Create3OperandInstr(MachineOpCode opCode,
                    Value* argVal1, MachineOperand::MachineOperandType type1,
                    Value* argVal2, MachineOperand::MachineOperandType type2,
                    Value* argVal3, MachineOperand::MachineOperandType type3)
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandVal(0, type1, argVal1);
  M->SetMachineOperandVal(1, type2, argVal2);
  M->SetMachineOperandVal(2, type3, argVal3);
  return M;
}

inline MachineInstr*
Create3OperandInstr(MachineOpCode opCode, Value* argVal1,
                    Value* argVal2, Value* argVal3)
{
  return Create3OperandInstr(opCode,
                             argVal1, MachineOperand::MO_VirtualRegister, 
                             argVal2, MachineOperand::MO_VirtualRegister, 
                             argVal3, MachineOperand::MO_VirtualRegister); 
}

inline MachineInstr*
Create3OperandInstr_UImmed(MachineOpCode opCode, Value* argVal1,
                           unsigned int unextendedImmed, Value* argVal3)
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister, argVal1);
  M->SetMachineOperandConst(1, MachineOperand::MO_UnextendedImmed,
                                 unextendedImmed);
  M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, argVal3);
  return M;
}

inline MachineInstr*
Create3OperandInstr_SImmed(MachineOpCode opCode, Value* argVal1,
                           int signExtendedImmed, Value* argVal3)
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister, argVal1);
  M->SetMachineOperandConst(1, MachineOperand::MO_SignExtendedImmed,
                                 signExtendedImmed);
  M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, argVal3);
  return M;
}

inline MachineInstr*
Create3OperandInstr_Addr(MachineOpCode opCode, Value* argVal1,
                         Value* label, Value* argVal3)
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister, argVal1);
  M->SetMachineOperandVal(1, MachineOperand::MO_PCRelativeDisp,  label);
  M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, argVal3);
  return M;
}

inline MachineInstr*
Create3OperandInstr_Reg(MachineOpCode opCode, Value* argVal1,
                        unsigned int regNum, Value* argVal3)
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister, argVal1);
  M->SetMachineOperandReg(1, regNum);
  M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, argVal3);
  return M;
}

inline MachineInstr*
Create3OperandInstr_Reg(MachineOpCode opCode, unsigned int regNum1,
                        unsigned int regNum2, Value* argVal3)
                 
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandReg(0, regNum1);
  M->SetMachineOperandReg(1, regNum2);
  M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, argVal3);
  return M;
}

inline MachineInstr*
Create3OperandInstr_Reg(MachineOpCode opCode, unsigned int regNum1,
                        unsigned int regNum2, unsigned int regNum3)
                 
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandReg(0, regNum1);
  M->SetMachineOperandReg(1, regNum2);
  M->SetMachineOperandReg(2, regNum3);
  return M;
}


//---------------------------------------------------------------------------
// Function: ChooseRegOrImmed
// 
// Purpose:
// 
//---------------------------------------------------------------------------

MachineOperand::MachineOperandType ChooseRegOrImmed(
                                         Value* val,
                                         MachineOpCode opCode,
                                         const TargetMachine& targetMachine,
                                         bool canUseImmed,
                                         unsigned int& getMachineRegNum,
                                         int64_t& getImmedValue);


//---------------------------------------------------------------------------
// Function: FixConstantOperandsForInstr
// 
// Purpose:
// Special handling for constant operands of a machine instruction
// -- if the constant is 0, use the hardwired 0 register, if any;
// -- if the constant fits in the IMMEDIATE field, use that field;
// -- else create instructions to put the constant into a register, either
//    directly or by loading explicitly from the constant pool.
// 
// In the first 2 cases, the operand of `minstr' is modified in place.
// Returns a vector of machine instructions generated for operands that
// fall under case 3; these must be inserted before `minstr'.
//---------------------------------------------------------------------------

std::vector<MachineInstr*> FixConstantOperandsForInstr (Instruction* vmInstr,
                                                        MachineInstr* minstr,
                                                        TargetMachine& target);

#endif
