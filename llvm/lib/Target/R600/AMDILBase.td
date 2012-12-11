//===- AMDIL.td - AMDIL Target Machine -------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

// Dummy Instruction itineraries for pseudo instructions
def ALU_NULL : FuncUnit;
def NullALU : InstrItinClass;

//===----------------------------------------------------------------------===//
// AMDIL Subtarget features.
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


//===----------------------------------------------------------------------===//
// Register File, Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//


include "AMDILRegisterInfo.td"
include "AMDILInstrInfo.td"

