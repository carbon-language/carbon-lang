//===- Mips.td - Describe the Mips Target Machine ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bruno Cardoso Lopes and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This is the top level entry point for the Mips target.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces
//===----------------------------------------------------------------------===//

include "../Target.td"

//===----------------------------------------------------------------------===//
// Descriptions
//===----------------------------------------------------------------------===//

include "MipsRegisterInfo.td"
include "MipsSchedule.td"
include "MipsInstrInfo.td"
include "MipsCallingConv.td"

def MipsInstrInfo : InstrInfo {
  let TSFlagsFields = [];
  let TSFlagsShifts = [];
}

//===----------------------------------------------------------------------===//
// CPU Directives                                                             //
//===----------------------------------------------------------------------===//

// Not currently supported, but work as SubtargetFeature placeholder.
def FeatureMipsIII : SubtargetFeature<"mips3", "IsMipsIII", "true",
                                      "MipsIII ISA Support">;

//===----------------------------------------------------------------------===//
// Mips processors supported.
//===----------------------------------------------------------------------===//

def : Processor<"mips1", MipsGenericItineraries, []>;
def : Processor<"r2000", MipsGenericItineraries, []>;
def : Processor<"r3000", MipsGenericItineraries, []>;

def Mips : Target {
  let InstructionSet = MipsInstrInfo;
}

