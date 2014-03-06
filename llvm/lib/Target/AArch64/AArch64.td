//===- AArch64.td - Describe the AArch64 Target Machine -------*- tblgen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the top level entry point for the AArch64 target.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// AArch64 Subtarget features.
//

def FeatureFPARMv8 : SubtargetFeature<"fp-armv8", "HasFPARMv8", "true",
  "Enable ARMv8 FP">;

def FeatureNEON : SubtargetFeature<"neon", "HasNEON", "true",
  "Enable Advanced SIMD instructions", [FeatureFPARMv8]>;

def FeatureCrypto : SubtargetFeature<"crypto", "HasCrypto", "true",
  "Enable cryptographic instructions">;

//===----------------------------------------------------------------------===//
// AArch64 Processors
//

include "AArch64Schedule.td"

class ProcNoItin<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

def : Processor<"generic", GenericItineraries, [FeatureFPARMv8, FeatureNEON]>;

def ProcA53     : SubtargetFeature<"a53", "ARMProcFamily", "CortexA53",
                                   "Cortex-A53 ARM processors",
                                   [FeatureFPARMv8,
                                   FeatureNEON,
                                   FeatureCrypto]>;

def ProcA57     : SubtargetFeature<"a57", "ARMProcFamily", "CortexA57",
                                   "Cortex-A57 ARM processors",
                                   [FeatureFPARMv8,
                                   FeatureNEON,
                                   FeatureCrypto]>;

def : ProcessorModel<"cortex-a53", CortexA53Model, [ProcA53]>;
def : Processor<"cortex-a57", NoItineraries, [ProcA57]>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "AArch64RegisterInfo.td"

include "AArch64CallingConv.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "AArch64InstrInfo.td"

def AArch64InstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def AArch64 : Target {
  let InstructionSet = AArch64InstrInfo;
}
