//===- Blackfin.td - Describe the Blackfin Target Machine --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Blackfin Subtarget features.
//===----------------------------------------------------------------------===//

def FeatureSSYNC : SubtargetFeature<"ssync","ssyncWorkaround", "true",
                                    "Work around SSYNC bugs">;

//===----------------------------------------------------------------------===//
// Register File, Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//

include "BlackfinRegisterInfo.td"
include "BlackfinCallingConv.td"
include "BlackfinInstrInfo.td"

def BlackfinInstrInfo : InstrInfo {}

//===----------------------------------------------------------------------===//
// Blackfin processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

def : Proc<"generic", [FeatureSSYNC]>;

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def Blackfin : Target {
  // Pull in Instruction Info:
  let InstructionSet = BlackfinInstrInfo;
}
