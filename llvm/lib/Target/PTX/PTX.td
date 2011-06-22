//===- PTX.td - Describe the PTX Target Machine ---------------*- tblgen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This is the top level entry point for the PTX target.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Subtarget Features
//===----------------------------------------------------------------------===//

//===- Architectural Features ---------------------------------------------===//

def FeatureDouble : SubtargetFeature<"double", "SupportsDouble", "true",
                                     "Do not demote .f64 to .f32">;

def FeatureNoFMA  : SubtargetFeature<"no-fma","SupportsFMA", "false",
                                     "Disable Fused-Multiply Add">;

//===- PTX Version --------------------------------------------------------===//

def FeaturePTX20 : SubtargetFeature<"ptx20", "PTXVersion", "PTX_VERSION_2_0",
                                    "Use PTX Language Version 2.0",
                                    []>;

def FeaturePTX21 : SubtargetFeature<"ptx21", "PTXVersion", "PTX_VERSION_2_1",
                                    "Use PTX Language Version 2.1",
                                    [FeaturePTX20]>;

def FeaturePTX22 : SubtargetFeature<"ptx22", "PTXVersion", "PTX_VERSION_2_2",
                                    "Use PTX Language Version 2.2",
                                    [FeaturePTX21]>;

def FeaturePTX23 : SubtargetFeature<"ptx23", "PTXVersion", "PTX_VERSION_2_3",
                                    "Use PTX Language Version 2.3",
                                    [FeaturePTX22]>;

//===- PTX Shader Model ---------------------------------------------------===//

def FeatureSM10 : SubtargetFeature<"sm10", "PTXShaderModel", "PTX_SM_1_0",
                                   "Enable Shader Model 1.0 compliance">;
def FeatureSM13 : SubtargetFeature<"sm13", "PTXShaderModel", "PTX_SM_1_3",
                                   "Enable Shader Model 1.3 compliance",
                                   [FeatureSM10, FeatureDouble]>;
def FeatureSM20 : SubtargetFeature<"sm20", "PTXShaderModel", "PTX_SM_2_0",
                                   "Enable Shader Model 2.0 compliance",
                                   [FeatureSM13]>;

//===----------------------------------------------------------------------===//
// PTX supported processors
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
  : Processor<Name, NoItineraries, Features>;

def : Proc<"generic", []>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "PTXRegisterInfo.td"

//===----------------------------------------------------------------------===//
// Calling Conventions
//===----------------------------------------------------------------------===//

include "PTXCallingConv.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "PTXInstrInfo.td"

def PTXInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Target Declaration
//===----------------------------------------------------------------------===//

def PTX : Target {
  let InstructionSet = PTXInstrInfo;
}
