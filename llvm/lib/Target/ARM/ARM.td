//===- ARM.td - Describe the ARM Target Machine -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

include "ARMRegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "ARMInstrInfo.td"

def ARMInstrInfo : InstrInfo {
  // Define how we want to layout our target-specific information field.
  let TSFlagsFields = [];
  let TSFlagsShifts = [];
}

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def ARM : Target {
  // FIXME: Specify callee-saved registers
  let CalleeSavedRegisters = [];

  // Pull in Instruction Info:
  let InstructionSet = ARMInstrInfo;
}
