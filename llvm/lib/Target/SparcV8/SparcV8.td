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
  let CalleeSavedRegisters = [];

  // Pull in Instruction Info:
  let InstructionSet = SparcV8InstrInfo;
}
