//===- ARM.td - Describe the ARM Target Machine -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// ARM Subtarget features.
//

def ArchV4T     : SubtargetFeature<"v4t", "ARMArchVersion", "V4T",
                                   "ARM v4T">;
def ArchV5T     : SubtargetFeature<"v5t", "ARMArchVersion", "V5T",
                                   "ARM v5T">;
def ArchV5TE    : SubtargetFeature<"v5te", "ARMArchVersion", "V5TE",
                                   "ARM v5TE, v5TEj, v5TExp">;
def ArchV6      : SubtargetFeature<"v6", "ARMArchVersion", "V6",
                                   "ARM v6">;
def ArchV6T2    : SubtargetFeature<"v6t2", "ARMArchVersion", "V6T2",
                                   "ARM v6t2">;
def ArchV7A     : SubtargetFeature<"v7a", "ARMArchVersion", "V7A",
                                   "ARM v7A">;
def FeatureVFP2 : SubtargetFeature<"vfp2", "ARMFPUType", "VFPv2",
                                   "Enable VFP2 instructions">;
def FeatureVFP3 : SubtargetFeature<"vfp3", "ARMFPUType", "VFPv3",
                                   "Enable VFP3 instructions">;
def FeatureNEON : SubtargetFeature<"neon", "ARMFPUType", "NEON",
                                   "Enable NEON instructions">;
def FeatureThumb2 : SubtargetFeature<"thumb2", "ThumbMode", "Thumb2",
                                     "Enable Thumb2 instructions">;

//===----------------------------------------------------------------------===//
// ARM Processors supported.
//

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

// V4 Processors.
def : Proc<"generic",         []>;
def : Proc<"arm8",            []>;
def : Proc<"arm810",          []>;
def : Proc<"strongarm",       []>;
def : Proc<"strongarm110",    []>;
def : Proc<"strongarm1100",   []>;
def : Proc<"strongarm1110",   []>;

// V4T Processors.
def : Proc<"arm7tdmi",        [ArchV4T]>;
def : Proc<"arm7tdmi-s",      [ArchV4T]>;
def : Proc<"arm710t",         [ArchV4T]>;
def : Proc<"arm720t",         [ArchV4T]>;
def : Proc<"arm9",            [ArchV4T]>;
def : Proc<"arm9tdmi",        [ArchV4T]>;
def : Proc<"arm920",          [ArchV4T]>;
def : Proc<"arm920t",         [ArchV4T]>;
def : Proc<"arm922t",         [ArchV4T]>;
def : Proc<"arm940t",         [ArchV4T]>;
def : Proc<"ep9312",          [ArchV4T]>;

// V5T Processors.
def : Proc<"arm10tdmi",       [ArchV5T]>;
def : Proc<"arm1020t",        [ArchV5T]>;

// V5TE Processors.
def : Proc<"arm9e",           [ArchV5TE]>;
def : Proc<"arm926ej-s",      [ArchV5TE]>;
def : Proc<"arm946e-s",       [ArchV5TE]>;
def : Proc<"arm966e-s",       [ArchV5TE]>;
def : Proc<"arm968e-s",       [ArchV5TE]>;
def : Proc<"arm10e",          [ArchV5TE]>;
def : Proc<"arm1020e",        [ArchV5TE]>;
def : Proc<"arm1022e",        [ArchV5TE]>;
def : Proc<"xscale",          [ArchV5TE]>;
def : Proc<"iwmmxt",          [ArchV5TE]>;

// V6 Processors.
def : Proc<"arm1136j-s",      [ArchV6]>;
def : Proc<"arm1136jf-s",     [ArchV6, FeatureVFP2]>;
def : Proc<"arm1176jz-s",     [ArchV6]>;
def : Proc<"arm1176jzf-s",    [ArchV6, FeatureVFP2]>;
def : Proc<"mpcorenovfp",     [ArchV6]>;
def : Proc<"mpcore",          [ArchV6, FeatureVFP2]>;

// V6T2 Processors.
def : Proc<"arm1156t2-s",     [ArchV6T2, FeatureThumb2]>;
def : Proc<"arm1156t2f-s",    [ArchV6T2, FeatureThumb2, FeatureVFP2]>;

// V7 Processors.
def : Proc<"cortex-a8",       [ArchV7A, FeatureThumb2, FeatureNEON]>;
def : Proc<"cortex-a9",       [ArchV7A, FeatureThumb2, FeatureNEON]>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "ARMRegisterInfo.td"

include "ARMCallingConv.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "ARMInstrInfo.td"

def ARMInstrInfo : InstrInfo {
  // Define how we want to layout our target-specific information field.
  let TSFlagsFields = ["AddrModeBits",
                       "SizeFlag",
                       "IndexModeBits",
                       "isUnaryDataProc",
                       "Form"];
  let TSFlagsShifts = [0,
                       4,
                       7,
                       9,
                       10];
}

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def ARM : Target {
  // Pull in Instruction Info:
  let InstructionSet = ARMInstrInfo;
}
