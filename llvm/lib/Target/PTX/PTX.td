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
                                    "Use PTX Language Version 2.0">;

def FeaturePTX21 : SubtargetFeature<"ptx21", "PTXVersion", "PTX_VERSION_2_1",
                                    "Use PTX Language Version 2.1">;

def FeaturePTX22 : SubtargetFeature<"ptx22", "PTXVersion", "PTX_VERSION_2_2",
                                    "Use PTX Language Version 2.2">;

def FeaturePTX23 : SubtargetFeature<"ptx23", "PTXVersion", "PTX_VERSION_2_3",
                                    "Use PTX Language Version 2.3">;

//===- PTX Target ---------------------------------------------------------===//

def FeatureSM10 : SubtargetFeature<"sm10", "PTXTarget", "PTX_SM_1_0",
                                   "Use Shader Model 1.0">;
def FeatureSM11 : SubtargetFeature<"sm11", "PTXTarget", "PTX_SM_1_1",
                                   "Use Shader Model 1.1">;
def FeatureSM12 : SubtargetFeature<"sm12", "PTXTarget", "PTX_SM_1_2",
                                   "Use Shader Model 1.2">;
def FeatureSM13 : SubtargetFeature<"sm13", "PTXTarget", "PTX_SM_1_3",
                                   "Use Shader Model 1.3">;
def FeatureSM20 : SubtargetFeature<"sm20", "PTXTarget", "PTX_SM_2_0",
                                   "Use Shader Model 2.0">;
def FeatureSM21 : SubtargetFeature<"sm21", "PTXTarget", "PTX_SM_2_1",
                                   "Use Shader Model 2.1">;
def FeatureSM22 : SubtargetFeature<"sm22", "PTXTarget", "PTX_SM_2_2",
                                   "Use Shader Model 2.2">;
def FeatureSM23 : SubtargetFeature<"sm23", "PTXTarget", "PTX_SM_2_3",
                                   "Use Shader Model 2.3">;

def FeatureCOMPUTE10 : SubtargetFeature<"compute10", "PTXTarget",
                                        "PTX_COMPUTE_1_0",
                                        "Use Compute Compatibility 1.0">;
def FeatureCOMPUTE11 : SubtargetFeature<"compute11", "PTXTarget",
                                        "PTX_COMPUTE_1_1",
                                        "Use Compute Compatibility 1.1">;
def FeatureCOMPUTE12 : SubtargetFeature<"compute12", "PTXTarget",
                                        "PTX_COMPUTE_1_2",
                                        "Use Compute Compatibility 1.2">;
def FeatureCOMPUTE13 : SubtargetFeature<"compute13", "PTXTarget",
                                        "PTX_COMPUTE_1_3",
                                        "Use Compute Compatibility 1.3">;
def FeatureCOMPUTE20 : SubtargetFeature<"compute20", "PTXTarget",
                                        "PTX_COMPUTE_2_0",
                                        "Use Compute Compatibility 2.0">;

//===----------------------------------------------------------------------===//
// PTX supported processors
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
  : Processor<Name, NoItineraries, Features>;

def : Proc<"generic", []>;

// Processor definitions for compute/shader models
def : Proc<"compute_10", [FeatureCOMPUTE10]>;
def : Proc<"compute_11", [FeatureCOMPUTE11]>;
def : Proc<"compute_12", [FeatureCOMPUTE12]>;
def : Proc<"compute_13", [FeatureCOMPUTE13]>;
def : Proc<"compute_20", [FeatureCOMPUTE20]>;
def : Proc<"sm_10",      [FeatureSM10]>;
def : Proc<"sm_11",      [FeatureSM11]>;
def : Proc<"sm_12",      [FeatureSM12]>;
def : Proc<"sm_13",      [FeatureSM13]>;
def : Proc<"sm_20",      [FeatureSM20]>;
def : Proc<"sm_21",      [FeatureSM21]>;
def : Proc<"sm_22",      [FeatureSM22]>;
def : Proc<"sm_23",      [FeatureSM23]>;

// Processor definitions for common GPU architectures
def : Proc<"g80",        [FeatureSM10]>;
def : Proc<"gt200",      [FeatureSM13]>;
def : Proc<"gf100",      [FeatureSM20, FeatureDouble]>;
def : Proc<"fermi",      [FeatureSM20, FeatureDouble]>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "PTXRegisterInfo.td"

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
