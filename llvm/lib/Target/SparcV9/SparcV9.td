//===- SparcV9.td - Target Description for SparcV9 Target --*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// TableGen target description file for the SparcV9. This is currently used 
// primarily to generate part of the SparcV9CodeEmitter automatically.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "../Target.td"

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "SparcV9RegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "SparcV9InstrInfo.td"

def SparcV9InstrInfo : InstrInfo {
  let PHIInst = PHI;

  // Define how we want to layout our TargetSpecific information field.
  let TSFlagsFields = [];
  let TSFlagsShifts = [];
}

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def SparcV9 : Target {
  // FIXME: Specify the callee saved registers.
  let CalleeSavedRegisters = [];

  // Pointers are 64-bits in size.
  let PointerType = i64;

  // Information about the instructions...
  let InstructionSet = SparcV9InstrInfo;
}
