//===-- llvm/CodeGen/InstrSelectionSupport.h --------------------*- C++ -*-===//
//
//  Target-independent instruction selection code.  See SparcInstrSelection.cpp
//  for usage.
//      
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_INSTR_SELECTION_SUPPORT_H
#define LLVM_CODEGEN_INSTR_SELECTION_SUPPORT_H

#include "llvm/CodeGen/MachineInstr.h"
#include "Support/DataTypes.h"
class InstructionNode;
class TargetMachine;
class Instruction;

//---------------------------------------------------------------------------
// Function GetConstantValueAsUnsignedInt
// Function GetConstantValueAsSignedInt
// 
// Convenience functions to get the value of an integer constant, for an
// appropriate integer or non-integer type that can be held in a signed
// or unsigned integer respectively.  The type of the argument must be
// the following:
//      Signed or unsigned integer
//      Boolean
//      Pointer
// 
// isValidConstant is set to true if a valid constant was found.
//---------------------------------------------------------------------------

uint64_t        GetConstantValueAsUnsignedInt   (const Value *V,
                                                 bool &isValidConstant);

int64_t         GetConstantValueAsSignedInt     (const Value *V,
                                                 bool &isValidConstant);


//---------------------------------------------------------------------------
// Function: GetMemInstArgs
// 
// Purpose:
//   Get the pointer value and the index vector for a memory operation
//   (GetElementPtr, Load, or Store).  If all indices of the given memory
//   operation are constant, fold in constant indices in a chain of
//   preceding GetElementPtr instructions (if any), and return the
//   pointer value of the first instruction in the chain.
//   All folded instructions are marked so no code is generated for them.
//
// Return values:
//   Returns the pointer Value to use.
//   Returns the resulting IndexVector in idxVec.
//   Returns true/false in allConstantIndices if all indices are/aren't const.
//---------------------------------------------------------------------------

Value*          GetMemInstArgs  (InstructionNode* memInstrNode,
                                 std::vector<Value*>& idxVec,
                                 bool& allConstantIndices);


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
// RETURN VALUE: unsigned flags, where
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
Create1OperandInstr_UImmed(MachineOpCode opCode, unsigned unextendedImmed)
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandConst(0, MachineOperand::MO_UnextendedImmed,
                               unextendedImmed);
  return M;
}

inline MachineInstr*
Create1OperandInstr_SImmed(MachineOpCode opCode, int signExtendedImmed)
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandConst(0, MachineOperand::MO_SignExtendedImmed,
                               signExtendedImmed);
  return M;
}

inline MachineInstr*
Create1OperandInstr_Addr(MachineOpCode opCode, Value* label)
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandVal(0, MachineOperand::MO_PCRelativeDisp, label);
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
                           unsigned unextendedImmed, Value* argVal2)
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
                        Value* argVal1, unsigned regNum)
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister, argVal1);
  M->SetMachineOperandReg(1, regNum);
  return M;
}

inline MachineInstr*
Create2OperandInstr_Reg(MachineOpCode opCode,
                        unsigned regNum1, unsigned regNum2)
                 
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandReg(0, regNum1);
  M->SetMachineOperandReg(1, regNum2);
  return M;
}

inline MachineInstr*
Create3OperandInstr_Reg(MachineOpCode opCode, Value* argVal1,
                        unsigned regNum, Value* argVal3)
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister, argVal1);
  M->SetMachineOperandReg(1, regNum);
  M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, argVal3);
  return M;
}

inline MachineInstr*
Create3OperandInstr_Reg(MachineOpCode opCode, unsigned regNum1,
                        unsigned regNum2, Value* argVal3)
                 
{
  MachineInstr* M = new MachineInstr(opCode);
  M->SetMachineOperandReg(0, regNum1);
  M->SetMachineOperandReg(1, regNum2);
  M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, argVal3);
  return M;
}

inline MachineInstr*
Create3OperandInstr_Reg(MachineOpCode opCode, unsigned regNum1,
                        unsigned regNum2, unsigned regNum3)
                 
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
                                         unsigned& getMachineRegNum,
                                         int64_t& getImmedValue);

MachineOperand::MachineOperandType ChooseRegOrImmed(int64_t intValue,
                                         bool isSigned,
                                         MachineOpCode opCode,
                                         const TargetMachine& target,
                                         bool canUseImmed,
                                         unsigned& getMachineRegNum,
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
