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

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "../Target.td"

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "SparcV8RegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "SparcV8InstrInfo.td"

def SparcV8InstrInfo : InstrInfo {
  let PHIInst = PHI;

  // Define how we want to layout our target-specific information field.
  let TSFlagsFields = [];
  let TSFlagsShifts = [];
}

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def SparcV8 : Target {
  // Pointers are 32-bits in size.
  let PointerType = i32;

  // FIXME: Specify callee-saved registers
  let CalleeSavedRegisters = [];

  // Pull in Instruction Info:
  let InstructionSet = SparcV8InstrInfo;
}
