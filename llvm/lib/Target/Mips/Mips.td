//===- Mips.td - Describe the Mips Target Machine ----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This is the top level entry point for the Mips target.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Register File, Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//

include "MipsRegisterInfo.td"
include "MipsSchedule.td"
include "MipsInstrInfo.td"
include "MipsCallingConv.td"

def MipsInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Mips Subtarget features                                                    //
//===----------------------------------------------------------------------===//

def FeatureGP64Bit     : SubtargetFeature<"gp64", "IsGP64bit", "true",
                                "General Purpose Registers are 64-bit wide.">;
def FeatureFP64Bit     : SubtargetFeature<"fp64", "IsFP64bit", "true",
                                "Support 64-bit FP registers.">;
def FeatureSingleFloat : SubtargetFeature<"single-float", "IsSingleFloat",
                                "true", "Only supports single precision float">;
def FeatureO32         : SubtargetFeature<"o32", "MipsABI", "O32",
                                "Enable o32 ABI">;
def FeatureEABI        : SubtargetFeature<"eabi", "MipsABI", "EABI",
                                "Enable eabi ABI">;
def FeatureVFPU        : SubtargetFeature<"vfpu", "HasVFPU",
                                "true", "Enable vector FPU instructions.">;
def FeatureSEInReg     : SubtargetFeature<"seinreg", "HasSEInReg", "true",
                                "Enable 'signext in register' instructions.">;
def FeatureCondMov     : SubtargetFeature<"condmov", "HasCondMov", "true",
                                "Enable 'conditional move' instructions.">;
def FeatureMulDivAdd   : SubtargetFeature<"muldivadd", "HasMulDivAdd", "true",
                                "Enable 'multiply add/sub' instructions.">;
def FeatureMinMax      : SubtargetFeature<"minmax", "HasMinMax", "true",
                                "Enable 'min/max' instructions.">;
def FeatureSwap        : SubtargetFeature<"swap", "HasSwap", "true",
                                "Enable 'byte/half swap' instructions.">;
def FeatureBitCount    : SubtargetFeature<"bitcount", "HasBitCount", "true",
                                "Enable 'count leading bits' instructions.">;
def FeatureMips32      : SubtargetFeature<"mips32", "MipsArchVersion", "Mips32",
                                "Mips32 ISA Support",
                                [FeatureCondMov, FeatureBitCount]>;
def FeatureMips32r2    : SubtargetFeature<"mips32r2", "MipsArchVersion",
                                "Mips32r2", "Mips32r2 ISA Support",
                                [FeatureMips32, FeatureSEInReg]>;

//===----------------------------------------------------------------------===//
// Mips processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, MipsGenericItineraries, Features>;

def : Proc<"mips32r1", [FeatureMips32]>;
def : Proc<"4ke", [FeatureMips32r2]>;

def MipsAsmWriter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  bit isMCAsmWriter = 1;
}

def Mips : Target {
  let InstructionSet = MipsInstrInfo;

  let AssemblyWriters = [MipsAsmWriter];
}

