//===- PowerPC.td - Describe the PowerPC Target Machine ----*- tablegen -*-===//
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

include "PowerPCRegisterInfo.td"
include "PowerPCInstrInfo.td"

def PowerPCInstrInfo : InstrInfo {
  let PHIInst  = PHI;

  let TSFlagsFields = ["ArgCount", "Arg0Type", "Arg1Type", "Arg2Type", 
                       "Arg3Type", "Arg4Type", "VMX", "PPC64"];
  let TSFlagsShifts = [ 0, 3, 8, 13, 18, 23, 28, 29 ];
}
