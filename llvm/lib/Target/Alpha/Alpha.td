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
// Subtarget Features
//===----------------------------------------------------------------------===//

def FeatureCIX : SubtargetFeature<"CIX", "HasCT", "true",
                                  "Enable CIX extentions">;
def FeatureFIX : SubtargetFeature<"FIX", "HasF2I", "true",
                                  "Enable FIX extentions">;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "AlphaRegisterInfo.td"

//===----------------------------------------------------------------------===//
// Schedule Description
//===----------------------------------------------------------------------===//

include "AlphaSchedule.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "AlphaInstrInfo.td"

def AlphaInstrInfo : InstrInfo {
  // Define how we want to layout our target-specific information field.
 // let TSFlagsFields = [];
 // let TSFlagsShifts = [];
}

//===----------------------------------------------------------------------===//
// Alpha Processor Definitions
//===----------------------------------------------------------------------===//

def : Processor<"generic", Alpha21264Itineraries, []>;
def : Processor<"pca56"  , Alpha21264Itineraries, []>;
def : Processor<"ev56"   , Alpha21264Itineraries, []>;
def : Processor<"ev6"    , Alpha21264Itineraries, [FeatureFIX]>;
def : Processor<"ev67"   , Alpha21264Itineraries, [FeatureFIX, FeatureCIX]>;

//===----------------------------------------------------------------------===//
// The Alpha Target
//===----------------------------------------------------------------------===//


def Alpha : Target {
  // Pull in Instruction Info:
  let InstructionSet = AlphaInstrInfo;
}
