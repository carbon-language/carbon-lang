//===- X86.td - Target definition file for the Intel X86 arch ---*- C++ -*-===//
//
// This is a target description file for the Intel i386 architecture, refered to
// here as the "X86" architecture.
//
//===----------------------------------------------------------------------===//

// Get the target independent interfaces which we are implementing...
//
include "../Target.td"

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "X86RegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

def X86InstrInfo : InstrInfo {
}

def X86 : Target {
  // Specify the callee saved registers.
  set CalleeSavedRegisters = [ESI, EDI, EBX, EBP];

  // Yes, pointers are 32-bits in size.
  set PointerType = i32;

  // Information about the instructions...
  set InstructionSet = X86InstrInfo;
}
