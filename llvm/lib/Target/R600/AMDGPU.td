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

// Target features

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

class SubtargetFeatureFetchLimit <string Value> :
                          SubtargetFeature <"fetch"#Value,
        "TexVTXClauseSize",
        Value,
        "Limit the maximum number of fetches in a clause to "#Value>;

def FeatureFetchLimit8 : SubtargetFeatureFetchLimit <"8">;
def FeatureFetchLimit16 : SubtargetFeatureFetchLimit <"16">;

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
        [FeatureFetchLimit16]>;

def FeatureSouthernIslands : SubtargetFeatureGeneration<"SOUTHERN_ISLANDS",
        [Feature64BitPtr, FeatureFP64]>;

//===----------------------------------------------------------------------===//

def AMDGPUInstrInfo : InstrInfo {
  let guessInstructionProperties = 1;
}

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//
def AMDGPUAsmWriter : AsmWriter {
    string AsmWriterClassName = "InstPrinter";
    int Variant = 0;
    bit isMCAsmWriter = 1;
}

def AMDGPU : Target {
  // Pull in Instruction Info:
  let InstructionSet = AMDGPUInstrInfo;
  let AssemblyWriters = [AMDGPUAsmWriter];
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
