//===- SparcV8.td - Describe the SparcV8 Target Machine ---------*- C++ -*-===//
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

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "../Target.td"

//===----------------------------------------------------------------------===//
// SPARC Subtarget features.
//
 
def FeatureV9
  : SubtargetFeature<"v9", "IsV9", "true",
                     "Enable SPARC-V9 instructions">;
def FeatureV8Deprecated
  : SubtargetFeature<"deprecated-v8", "V8DeprecatedInsts", "true",
                     "Enable deprecated V8 instructions in V9 mode">;
def FeatureVIS
  : SubtargetFeature<"vis", "IsVIS", "true",
                     "Enable UltraSPARC Visual Instruction Set extensions">;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "SparcV8RegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "SparcV8InstrInfo.td"

def SparcV8InstrInfo : InstrInfo {
  // Define how we want to layout our target-specific information field.
  let TSFlagsFields = [];
  let TSFlagsShifts = [];
}

//===----------------------------------------------------------------------===//
// SPARC processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

def : Proc<"generic",         []>;
def : Proc<"v8",              []>;
def : Proc<"supersparc",      []>;
def : Proc<"sparclite",       []>;
def : Proc<"f934",            []>;
def : Proc<"hypersparc",      []>;
def : Proc<"sparclite86x",    []>;
def : Proc<"sparclet",        []>;
def : Proc<"tsc701",          []>;
def : Proc<"v9",              [FeatureV9]>;
def : Proc<"ultrasparc",      [FeatureV9, FeatureV8Deprecated]>;
def : Proc<"ultrasparc3",     [FeatureV9, FeatureV8Deprecated]>;
def : Proc<"ultrasparc3-vis", [FeatureV9, FeatureV8Deprecated, FeatureVIS]>;


//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def SparcV8 : Target {
  // Pointers are 32-bits in size.
  let PointerType = i32;

  // FIXME: Specify callee-saved registers
  let CalleeSavedRegisters = [];

  // Pull in Instruction Info:
  let InstructionSet = SparcV8InstrInfo;
}
