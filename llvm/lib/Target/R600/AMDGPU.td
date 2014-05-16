//===-- AMDIL.td - AMDIL Tablegen files --*- tablegen -*-------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//

// Include AMDIL TD files
include "AMDILBase.td"

//===----------------------------------------------------------------------===//
// Subtarget Features
//===----------------------------------------------------------------------===//

// Debugging Features

def FeatureDumpCode : SubtargetFeature <"DumpCode",
        "DumpCode",
        "true",
        "Dump MachineInstrs in the CodeEmitter">;

def FeatureIRStructurizer : SubtargetFeature <"disable-irstructurizer",
        "EnableIRStructurizer",
        "false",
        "Disable IR Structurizer">;

// Target features

def FeatureIfCvt : SubtargetFeature <"disable-ifcvt",
        "EnableIfCvt",
        "false",
        "Disable the if conversion pass">;

def FeatureFP64     : SubtargetFeature<"fp64",
        "FP64",
        "true",
        "Enable 64bit double precision operations">;

def Feature64BitPtr : SubtargetFeature<"64BitPtr",
        "Is64bit",
        "true",
        "Specify if 64bit addressing should be used.">;

def Feature32on64BitPtr : SubtargetFeature<"64on32BitPtr",
        "Is32on64bit",
        "false",
        "Specify if 64bit sized pointers with 32bit addressing should be used.">;

def FeatureR600ALUInst : SubtargetFeature<"R600ALUInst",
        "R600ALUInst",
        "false",
        "Older version of ALU instructions encoding.">;

def FeatureVertexCache : SubtargetFeature<"HasVertexCache",
        "HasVertexCache",
        "true",
        "Specify use of dedicated vertex cache.">;

def FeatureCaymanISA : SubtargetFeature<"caymanISA",
        "CaymanISA",
        "true",
        "Use Cayman ISA">;

def FeatureCFALUBug : SubtargetFeature<"cfalubug",
        "CFALUBug",
        "true",
        "GPU has CF_ALU bug">;

class SubtargetFeatureFetchLimit <string Value> :
                          SubtargetFeature <"fetch"#Value,
        "TexVTXClauseSize",
        Value,
        "Limit the maximum number of fetches in a clause to "#Value>;

def FeatureFetchLimit8 : SubtargetFeatureFetchLimit <"8">;
def FeatureFetchLimit16 : SubtargetFeatureFetchLimit <"16">;

class SubtargetFeatureWavefrontSize <int Value> : SubtargetFeature<
        "wavefrontsize"#Value,
        "WavefrontSize",
        !cast<string>(Value),
        "The number of threads per wavefront">;

def FeatureWavefrontSize16 : SubtargetFeatureWavefrontSize<16>;
def FeatureWavefrontSize32 : SubtargetFeatureWavefrontSize<32>;
def FeatureWavefrontSize64 : SubtargetFeatureWavefrontSize<64>;

class SubtargetFeatureGeneration <string Value,
                                  list<SubtargetFeature> Implies> :
        SubtargetFeature <Value, "Gen", "AMDGPUSubtarget::"#Value,
                          Value#" GPU generation", Implies>;

def FeatureR600 : SubtargetFeatureGeneration<"R600",
        [FeatureR600ALUInst, FeatureFetchLimit8]>;

def FeatureR700 : SubtargetFeatureGeneration<"R700",
        [FeatureFetchLimit16]>;

def FeatureEvergreen : SubtargetFeatureGeneration<"EVERGREEN",
        [FeatureFetchLimit16]>;

def FeatureNorthernIslands : SubtargetFeatureGeneration<"NORTHERN_ISLANDS",
        [FeatureFetchLimit16, FeatureWavefrontSize64]>;

def FeatureSouthernIslands : SubtargetFeatureGeneration<"SOUTHERN_ISLANDS",
        [Feature64BitPtr, FeatureFP64]>;

def FeatureSeaIslands : SubtargetFeatureGeneration<"SEA_ISLANDS",
        [Feature64BitPtr, FeatureFP64]>;
//===----------------------------------------------------------------------===//

def AMDGPUInstrInfo : InstrInfo {
  let guessInstructionProperties = 1;
}

def AMDGPU : Target {
  // Pull in Instruction Info:
  let InstructionSet = AMDGPUInstrInfo;
}

//===----------------------------------------------------------------------===//
// Predicate helper class
//===----------------------------------------------------------------------===//

class PredicateControl {
  Predicate SubtargetPredicate;
  list<Predicate> OtherPredicates = [];
  list<Predicate> Predicates = !listconcat([SubtargetPredicate],
                                            OtherPredicates);
}

// Include AMDGPU TD files
include "R600Schedule.td"
include "SISchedule.td"
include "Processors.td"
include "AMDGPUInstrInfo.td"
include "AMDGPUIntrinsics.td"
include "AMDGPURegisterInfo.td"
include "AMDGPUInstructions.td"
include "AMDGPUCallingConv.td"
