//===- X86.td - Target definition file for the Intel X86 arch ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This is a target description file for the Intel i386 architecture, refered to
// here as the "X86" architecture.
//
//===----------------------------------------------------------------------===//

// Get the target-independent interfaces which we are implementing...
//
include "../Target.td"

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "X86RegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "X86InstrInfo.td"

def X86InstrInfo : InstrInfo {
  let PHIInst  = PHI;

  // Define how we want to layout our TargetSpecific information field... This
  // should be kept up-to-date with the fields in the X86InstrInfo.h file.
  let TSFlagsFields = ["FormBits"  , "hasOpSizePrefix" ,  "Prefix", "TypeBits",
                       "FPFormBits", "printImplicitUses", "Opcode"];
  let TSFlagsShifts = [           0,                   5,        6,         10,
                                 13,                  16,       17];
}

def X86 : Target {
  // Specify the callee saved registers.
  let CalleeSavedRegisters = [ESI, EDI, EBX, EBP];

  // Yes, pointers are 32-bits in size.
  let PointerType = i32;

  // Information about the instructions...
  let InstructionSet = X86InstrInfo;
}
