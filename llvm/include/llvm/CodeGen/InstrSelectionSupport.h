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
