//===- SparcV8.td - Describe the SparcV8 Target Machine ---------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

// Get the target-independent interfaces which we are implementing...
//
include "../Target.td"

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "SparcV8Reg.td"
include "SparcV8Instrs.td"

def SparcV8InstrInfo : InstrInfo {
  let PHIInst  = PHI;
}

def SparcV8 : Target {
  // Pointers are 32-bits in size.
  let PointerType = i32;

  // According to the Mach-O Runtime ABI, these regs are nonvolatile across
  // calls:
  let CalleeSavedRegisters = [R1, R13, R14, R15, R16, R17, R18, R19,
    R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31, F14, F15,
    F16, F17, F18, F19, F20, F21, F22, F23, F24, F25, F26, F27, F28, F29,
    F30, F31, CR2, CR3, CR4];

  // Pull in Instruction Info:
  let InstructionSet = SparcV8InstrInfo;
}
