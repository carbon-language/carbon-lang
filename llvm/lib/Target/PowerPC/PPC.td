//===- PPC.td - Describe the PowerPC Target Machine --------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This is the top level entry point for the PowerPC target.
//
//===----------------------------------------------------------------------===//

// Get the target-independent interfaces which we are implementing.
//
include "../Target.td"

//===----------------------------------------------------------------------===//
// PowerPC Subtarget features.
//
 
def Feature64Bit     : SubtargetFeature<"64bit","Is64Bit", "true",
                                        "Enable 64-bit instructions">;
def Feature64BitRegs : SubtargetFeature<"64bitregs","Has64BitRegs", "true",
                                        "Enable 64-bit registers [beta]">;
def FeatureAltivec   : SubtargetFeature<"altivec","HasAltivec", "true",
                                        "Enable Altivec instructions">;
def FeatureGPUL      : SubtargetFeature<"gpul","IsGigaProcessor", "true",
                                        "Enable GPUL instructions">;
def FeatureFSqrt     : SubtargetFeature<"fsqrt","HasFSQRT", "true",
                                        "Enable the fsqrt instruction">; 
def FeatureSTFIWX    : SubtargetFeature<"stfiwx","HasSTFIWX", "true",
                                        "Enable the stfiwx instruction">; 

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "PPCRegisterInfo.td"
include "PPCSchedule.td"
include "PPCInstrInfo.td"

//===----------------------------------------------------------------------===//
// PowerPC processors supported.
//

def : Processor<"generic", G3Itineraries, []>;
def : Processor<"601", G3Itineraries, []>;
def : Processor<"602", G3Itineraries, []>;
def : Processor<"603", G3Itineraries, []>;
def : Processor<"603e", G3Itineraries, []>;
def : Processor<"603ev", G3Itineraries, []>;
def : Processor<"604", G3Itineraries, []>;
def : Processor<"604e", G3Itineraries, []>;
def : Processor<"620", G3Itineraries, []>;
def : Processor<"g3", G3Itineraries, []>;
def : Processor<"7400", G4Itineraries, [FeatureAltivec]>;
def : Processor<"g4", G4Itineraries, [FeatureAltivec]>;
def : Processor<"7450", G4PlusItineraries, [FeatureAltivec]>;
def : Processor<"g4+", G4PlusItineraries, [FeatureAltivec]>;
def : Processor<"750", G3Itineraries, []>;
def : Processor<"970", G5Itineraries,
                  [FeatureAltivec, FeatureGPUL, FeatureFSqrt, FeatureSTFIWX,
                   Feature64Bit /*, Feature64BitRegs */]>;
def : Processor<"g5", G5Itineraries,
                  [FeatureAltivec, FeatureGPUL, FeatureFSqrt, FeatureSTFIWX,
                   Feature64Bit /*, Feature64BitRegs */]>;


def PPCInstrInfo : InstrInfo {
  // Define how we want to layout our TargetSpecific information field... This
  // should be kept up-to-date with the fields in the PPCInstrInfo.h file.
  let TSFlagsFields = ["PPC970_First",
                       "PPC970_Single",
                       "PPC970_Cracked",
                       "PPC970_Unit"];
  let TSFlagsShifts = [0, 1, 2, 3];

  let isLittleEndianEncoding = 1;
}


def PPC : Target {
  // Pointers on PPC are 32-bits in size.
  let PointerType = i32;

  // Information about the instructions.
  let InstructionSet = PPCInstrInfo;


  // According to the Mach-O Runtime ABI, these regs are nonvolatile across
  // calls
  let CalleeSavedRegisters = [R1, R13, R14, R15, R16, R17, R18, R19,
    R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31, F14, F15,
    F16, F17, F18, F19, F20, F21, F22, F23, F24, F25, F26, F27, F28, F29,
    F30, F31, CR2, CR3, CR4, LR];
}
