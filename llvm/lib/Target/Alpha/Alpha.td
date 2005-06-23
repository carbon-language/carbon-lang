//===- Alpha.td - Describe the Alpha Target Machine --------*- tablegen -*-===//
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

//Alpha is little endian

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "AlphaRegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "AlphaInstrInfo.td"

def AlphaInstrInfo : InstrInfo {
  let PHIInst = PHI;

  // Define how we want to layout our target-specific information field.
 // let TSFlagsFields = [];
 // let TSFlagsShifts = [];
}

def Alpha : Target {
  // Pointers on Alpha are 64-bits in size.
  let PointerType = i64;

  let CalleeSavedRegisters = 
	//saved regs
	[R9, R10, R11, R12, R13, R14, 
	//Frame pointer
//	R15, 
	//return address
//	R26, 
	//Stack Pointer
//	R30,
         F2, F3, F4, F5, F6, F7, F8, F9];

  // Pull in Instruction Info:
  let InstructionSet = AlphaInstrInfo;
}
