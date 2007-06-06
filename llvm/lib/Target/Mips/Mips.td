//===- Mips.td - Describe the Mips Target Machine ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bruno Cardoso Lopes and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "../Target.td"

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "MipsRegisterInfo.td"

//===----------------------------------------------------------------------===//
// Subtarget features
//===----------------------------------------------------------------------===//

// TODO: dummy, needed to compile
def FeatureCIX : SubtargetFeature<"r3000", "isR3000", "true",
                                  "Enable r3000 extentions">;

//===----------------------------------------------------------------------===//
// Instruction Description
//===----------------------------------------------------------------------===//

include "MipsInstrInfo.td"

def MipsInstrInfo : InstrInfo {
  // Define how we want to layout our target-specific information field.
  let TSFlagsFields = [];
  let TSFlagsShifts = [];
}
//===----------------------------------------------------------------------===//
// Calling Conventions
//===----------------------------------------------------------------------===//

include "MipsCallingConv.td"

//===----------------------------------------------------------------------===//
// Mips processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

def : Proc<"generic", []>;

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def Mips : Target {
  // Pull in Instruction Info:
  let InstructionSet = MipsInstrInfo;
}
