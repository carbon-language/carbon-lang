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

#endif
