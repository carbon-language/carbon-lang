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

def FeatureFP64     : SubtargetFeature<"fp64",
        "CapsOverride[AMDGPUDeviceInfo::DoubleOps]",
        "true",
        "Enable 64bit double precision operations">;
def FeatureByteAddress    : SubtargetFeature<"byte_addressable_store",
        "CapsOverride[AMDGPUDeviceInfo::ByteStores]",
        "true",
        "Enable byte addressable stores">;
def FeatureBarrierDetect : SubtargetFeature<"barrier_detect",
        "CapsOverride[AMDGPUDeviceInfo::BarrierDetect]",
        "true",
        "Enable duplicate barrier detection(HD5XXX or later).">;
def FeatureImages : SubtargetFeature<"images",
        "CapsOverride[AMDGPUDeviceInfo::Images]",
        "true",
        "Enable image functions">;
def FeatureMultiUAV : SubtargetFeature<"multi_uav",
        "CapsOverride[AMDGPUDeviceInfo::MultiUAV]",
        "true",
        "Generate multiple UAV code(HD5XXX family or later)">;
def FeatureMacroDB : SubtargetFeature<"macrodb",
        "CapsOverride[AMDGPUDeviceInfo::MacroDB]",
        "true",
        "Use internal macrodb, instead of macrodb in driver">;
def FeatureNoAlias : SubtargetFeature<"noalias",
        "CapsOverride[AMDGPUDeviceInfo::NoAlias]",
        "true",
        "assert that all kernel argument pointers are not aliased">;
def FeatureNoInline : SubtargetFeature<"no-inline",
        "CapsOverride[AMDGPUDeviceInfo::NoInline]",
        "true",
        "specify whether to not inline functions">;

def Feature64BitPtr : SubtargetFeature<"64BitPtr",
        "Is64bit",
        "false",
        "Specify if 64bit addressing should be used.">;

def Feature32on64BitPtr : SubtargetFeature<"64on32BitPtr",
        "Is32on64bit",
        "false",
        "Specify if 64bit sized pointers with 32bit addressing should be used.">;
def FeatureDebug : SubtargetFeature<"debug",
        "CapsOverride[AMDGPUDeviceInfo::Debug]",
        "true",
        "Debug mode is enabled, so disable hardware accelerated address spaces.">;
def FeatureDumpCode : SubtargetFeature <"DumpCode",
        "DumpCode",
        "true",
        "Dump MachineInstrs in the CodeEmitter">;

def FeatureR600ALUInst : SubtargetFeature<"R600ALUInst",
        "R600ALUInst",
        "false",
        "Older version of ALU instructions encoding.">;

def FeatureVertexCache : SubtargetFeature<"HasVertexCache",
        "HasVertexCache",
        "true",
        "Specify use of dedicated vertex cache.">;

class SubtargetFeatureFetchLimit <string Value> :
                          SubtargetFeature <"fetch"#Value,
        "TexVTXClauseSize",
        Value,
        "Limit the maximum number of fetches in a clause to "#Value>;

def FeatureFetchLimit8 : SubtargetFeatureFetchLimit <"8">;
def FeatureFetchLimit16 : SubtargetFeatureFetchLimit <"16">;

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
