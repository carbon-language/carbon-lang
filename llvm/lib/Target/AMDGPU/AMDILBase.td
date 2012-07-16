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
        "CapsOverride[AMDILDeviceInfo::DoubleOps]",
        "true",
        "Enable 64bit double precision operations">;
def FeatureByteAddress    : SubtargetFeature<"byte_addressable_store",
        "CapsOverride[AMDILDeviceInfo::ByteStores]",
        "true",
        "Enable byte addressable stores">;
def FeatureBarrierDetect : SubtargetFeature<"barrier_detect",
        "CapsOverride[AMDILDeviceInfo::BarrierDetect]",
        "true",
        "Enable duplicate barrier detection(HD5XXX or later).">;
def FeatureImages : SubtargetFeature<"images",
        "CapsOverride[AMDILDeviceInfo::Images]",
        "true",
        "Enable image functions">;
def FeatureMultiUAV : SubtargetFeature<"multi_uav",
        "CapsOverride[AMDILDeviceInfo::MultiUAV]",
        "true",
        "Generate multiple UAV code(HD5XXX family or later)">;
def FeatureMacroDB : SubtargetFeature<"macrodb",
        "CapsOverride[AMDILDeviceInfo::MacroDB]",
        "true",
        "Use internal macrodb, instead of macrodb in driver">;
def FeatureNoAlias : SubtargetFeature<"noalias",
        "CapsOverride[AMDILDeviceInfo::NoAlias]",
        "true",
        "assert that all kernel argument pointers are not aliased">;
def FeatureNoInline : SubtargetFeature<"no-inline",
        "CapsOverride[AMDILDeviceInfo::NoInline]",
        "true",
        "specify whether to not inline functions">;

def Feature64BitPtr : SubtargetFeature<"64BitPtr",
        "mIs64bit",
        "false",
        "Specify if 64bit addressing should be used.">;

def Feature32on64BitPtr : SubtargetFeature<"64on32BitPtr",
        "mIs32on64bit",
        "false",
        "Specify if 64bit sized pointers with 32bit addressing should be used.">;
def FeatureDebug : SubtargetFeature<"debug",
        "CapsOverride[AMDILDeviceInfo::Debug]",
        "true",
        "Debug mode is enabled, so disable hardware accelerated address spaces.">;
def FeatureDumpCode : SubtargetFeature <"DumpCode",
        "mDumpCode",
        "true",
        "Dump MachineInstrs in the CodeEmitter">;


//===----------------------------------------------------------------------===//
// Register File, Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//


include "AMDILRegisterInfo.td"
include "AMDILCallingConv.td"
include "AMDILInstrInfo.td"

def AMDILInstrInfo : InstrInfo {}

//===----------------------------------------------------------------------===//
// AMDIL processors supported.
//===----------------------------------------------------------------------===//
//include "Processors.td"

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//
def AMDILAsmWriter : AsmWriter {
    string AsmWriterClassName = "AsmPrinter";
    int Variant = 0;
}

def AMDILAsmParser : AsmParser {
    string AsmParserClassName = "AsmParser";
    int Variant = 0;

    string CommentDelimiter = ";";

    string RegisterPrefix = "r";

}


def AMDIL : Target {
  // Pull in Instruction Info:
  let InstructionSet = AMDILInstrInfo;
  let AssemblyWriters = [AMDILAsmWriter];
  let AssemblyParsers = [AMDILAsmParser];
}
