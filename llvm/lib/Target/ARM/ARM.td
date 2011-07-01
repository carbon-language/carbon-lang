//===- ARM.td - Describe the ARM Target Machine ------------*- tablegen -*-===//
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

def FeatureVFP2 : SubtargetFeature<"vfp2", "ARMFPUType", "VFPv2",
                                   "Enable VFP2 instructions">;
def FeatureVFP3 : SubtargetFeature<"vfp3", "ARMFPUType", "VFPv3",
                                   "Enable VFP3 instructions">;
def FeatureNEON : SubtargetFeature<"neon", "ARMFPUType", "NEON",
                                   "Enable NEON instructions">;
def FeatureThumb2 : SubtargetFeature<"thumb2", "ThumbMode", "Thumb2",
                                     "Enable Thumb2 instructions">;
def FeatureNoARM  : SubtargetFeature<"noarm", "NoARM", "true",
                                     "Does not support ARM mode execution">;
def FeatureFP16   : SubtargetFeature<"fp16", "HasFP16", "true",
                                     "Enable half-precision floating point">;
def FeatureD16    : SubtargetFeature<"d16", "HasD16", "true",
                                     "Restrict VFP3 to 16 double registers">;
def FeatureHWDiv  : SubtargetFeature<"hwdiv", "HasHardwareDivide", "true",
                                     "Enable divide instructions">;
def FeatureT2XtPk : SubtargetFeature<"t2xtpk", "HasT2ExtractPack", "true",
                                 "Enable Thumb2 extract and pack instructions">;
def FeatureDB     : SubtargetFeature<"db", "HasDataBarrier", "true",
                                   "Has data barrier (dmb / dsb) instructions">;
def FeatureSlowFPBrcc : SubtargetFeature<"slow-fp-brcc", "SlowFPBrcc", "true",
                                         "FP compare + branch is slow">;
def FeatureVFPOnlySP : SubtargetFeature<"fp-only-sp", "FPOnlySP", "true",
                          "Floating point unit supports single precision only">;

// Some processors have FP multiply-accumulate instructions that don't
// play nicely with other VFP / NEON instructions, and it's generally better
// to just not use them.
def FeatureHasSlowFPVMLx : SubtargetFeature<"slowfpvmlx", "SlowFPVMLx", "true",
                                         "Disable VFP / NEON MAC instructions">;

// Cortex-A8 / A9 Advanced SIMD has multiplier accumulator forwarding.
def FeatureVMLxForwarding : SubtargetFeature<"vmlx-forwarding",
                                       "HasVMLxForwarding", "true",
                                       "Has multiplier accumulator forwarding">;

// Some processors benefit from using NEON instructions for scalar
// single-precision FP operations.
def FeatureNEONForFP : SubtargetFeature<"neonfp", "UseNEONForSinglePrecisionFP",
                                        "true",
                                        "Use NEON for single precision FP">;

// Disable 32-bit to 16-bit narrowing for experimentation.
def FeaturePref32BitThumb : SubtargetFeature<"32bit", "Pref32BitThumb", "true",
                                             "Prefer 32-bit Thumb instrs">;

/// Some instructions update CPSR partially, which can add false dependency for
/// out-of-order implementation, e.g. Cortex-A9, unless each individual bit is
/// mapped to a separate physical register. Avoid partial CPSR update for these
/// processors.
def FeatureAvoidPartialCPSR : SubtargetFeature<"avoid-partial-cpsr",
                                               "AvoidCPSRPartialUpdate", "true",
                                 "Avoid CPSR partial update for OOO execution">;

/// Some M architectures don't have the DSP extension (v7E-M vs. v7M)
def FeatureDSPThumb2 : SubtargetFeature<"t2dsp", "Thumb2DSP", "true",
                                 "Supports v7 DSP instructions in Thumb2.">;

// Multiprocessing extension.
def FeatureMP : SubtargetFeature<"mp", "HasMPExtension", "true",
                                 "Supports Multiprocessing extension">;

// ARM architectures.
def ArchV4T     : SubtargetFeature<"v4t", "ARMArchVersion", "V4T",
                                   "ARM v4T">;
def ArchV5T     : SubtargetFeature<"v5t", "ARMArchVersion", "V5T",
                                   "ARM v5T">;
def ArchV5TE    : SubtargetFeature<"v5te", "ARMArchVersion", "V5TE",
                                   "ARM v5TE, v5TEj, v5TExp">;
def ArchV6      : SubtargetFeature<"v6", "ARMArchVersion", "V6",
                                   "ARM v6">;
def ArchV6M     : SubtargetFeature<"v6m", "ARMArchVersion", "V6M",
                                   "ARM v6m",
                                   [FeatureNoARM, FeatureDB]>;
def ArchV6T2    : SubtargetFeature<"v6t2", "ARMArchVersion", "V6T2",
                                   "ARM v6t2",
                                   [FeatureThumb2, FeatureDSPThumb2]>;
def ArchV7A     : SubtargetFeature<"v7a", "ARMArchVersion", "V7A",
                                   "ARM v7A",
                                   [FeatureThumb2, FeatureNEON, FeatureDB,
                                    FeatureDSPThumb2]>;
def ArchV7M     : SubtargetFeature<"v7m", "ARMArchVersion", "V7M",
                                   "ARM v7M",
                                   [FeatureThumb2, FeatureNoARM, FeatureDB,
                                    FeatureHWDiv]>;
def ArchV7EM    : SubtargetFeature<"v7em", "ARMArchVersion", "V7EM",
                                   "ARM v7E-M",
                                   [FeatureThumb2, FeatureNoARM, FeatureDB,
                                    FeatureHWDiv, FeatureDSPThumb2,
                                    FeatureT2XtPk]>;

//===----------------------------------------------------------------------===//
// ARM Processors supported.
//

include "ARMSchedule.td"

// ARM processor families.
def ProcOthers  : SubtargetFeature<"others", "ARMProcFamily", "Others",
                                   "One of the other ARM processor families">;
def ProcA8      : SubtargetFeature<"a8", "ARMProcFamily", "CortexA8",
                                   "Cortex-A8 ARM processors",
                                   [FeatureSlowFPBrcc, FeatureNEONForFP,
                                    FeatureHasSlowFPVMLx, FeatureVMLxForwarding,
                                    FeatureT2XtPk]>;
def ProcA9      : SubtargetFeature<"a9", "ARMProcFamily", "CortexA9",
                                   "Cortex-A9 ARM processors",
                                   [FeatureVMLxForwarding,
                                    FeatureT2XtPk, FeatureFP16,
                                    FeatureAvoidPartialCPSR]>;

class ProcNoItin<string Name, list<SubtargetFeature> Features>
 : Processor<Name, GenericItineraries, Features>;

// V4 Processors.
def : ProcNoItin<"generic",         []>;
def : ProcNoItin<"arm8",            []>;
def : ProcNoItin<"arm810",          []>;
def : ProcNoItin<"strongarm",       []>;
def : ProcNoItin<"strongarm110",    []>;
def : ProcNoItin<"strongarm1100",   []>;
def : ProcNoItin<"strongarm1110",   []>;

// V4T Processors.
def : ProcNoItin<"arm7tdmi",        [ArchV4T]>;
def : ProcNoItin<"arm7tdmi-s",      [ArchV4T]>;
def : ProcNoItin<"arm710t",         [ArchV4T]>;
def : ProcNoItin<"arm720t",         [ArchV4T]>;
def : ProcNoItin<"arm9",            [ArchV4T]>;
def : ProcNoItin<"arm9tdmi",        [ArchV4T]>;
def : ProcNoItin<"arm920",          [ArchV4T]>;
def : ProcNoItin<"arm920t",         [ArchV4T]>;
def : ProcNoItin<"arm922t",         [ArchV4T]>;
def : ProcNoItin<"arm940t",         [ArchV4T]>;
def : ProcNoItin<"ep9312",          [ArchV4T]>;

// V5T Processors.
def : ProcNoItin<"arm10tdmi",       [ArchV5T]>;
def : ProcNoItin<"arm1020t",        [ArchV5T]>;

// V5TE Processors.
def : ProcNoItin<"arm9e",           [ArchV5TE]>;
def : ProcNoItin<"arm926ej-s",      [ArchV5TE]>;
def : ProcNoItin<"arm946e-s",       [ArchV5TE]>;
def : ProcNoItin<"arm966e-s",       [ArchV5TE]>;
def : ProcNoItin<"arm968e-s",       [ArchV5TE]>;
def : ProcNoItin<"arm10e",          [ArchV5TE]>;
def : ProcNoItin<"arm1020e",        [ArchV5TE]>;
def : ProcNoItin<"arm1022e",        [ArchV5TE]>;
def : ProcNoItin<"xscale",          [ArchV5TE]>;
def : ProcNoItin<"iwmmxt",          [ArchV5TE]>;

// V6 Processors.
def : Processor<"arm1136j-s",       ARMV6Itineraries, [ArchV6]>;
def : Processor<"arm1136jf-s",      ARMV6Itineraries, [ArchV6, FeatureVFP2,
                                                       FeatureHasSlowFPVMLx]>;
def : Processor<"arm1176jz-s",      ARMV6Itineraries, [ArchV6]>;
def : Processor<"arm1176jzf-s",     ARMV6Itineraries, [ArchV6, FeatureVFP2,
                                                       FeatureHasSlowFPVMLx]>;
def : Processor<"mpcorenovfp",      ARMV6Itineraries, [ArchV6]>;
def : Processor<"mpcore",           ARMV6Itineraries, [ArchV6, FeatureVFP2,
                                                       FeatureHasSlowFPVMLx]>;

// V6M Processors.
def : Processor<"cortex-m0",        ARMV6Itineraries, [ArchV6M]>;

// V6T2 Processors.
def : Processor<"arm1156t2-s",      ARMV6Itineraries, [ArchV6T2]>;
def : Processor<"arm1156t2f-s",     ARMV6Itineraries, [ArchV6T2, FeatureVFP2,
                                                       FeatureHasSlowFPVMLx]>;

// V7 Processors.
def : Processor<"cortex-a8",        CortexA8Itineraries,
                                    [ArchV7A, ProcA8]>;
def : Processor<"cortex-a9",        CortexA9Itineraries,
                                    [ArchV7A, ProcA9]>;
def : Processor<"cortex-a9-mp",     CortexA9Itineraries,
                                    [ArchV7A, ProcA9, FeatureMP]>;

// V7M Processors.
def : ProcNoItin<"cortex-m3",       [ArchV7M]>;
def : ProcNoItin<"cortex-m4",       [ArchV7EM, FeatureVFP2, FeatureVFPOnlySP]>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "ARMRegisterInfo.td"

include "ARMCallingConv.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "ARMInstrInfo.td"

def ARMInstrInfo : InstrInfo;


//===----------------------------------------------------------------------===//
// Assembly printer
//===----------------------------------------------------------------------===//
// ARM Uses the MC printer for asm output, so make sure the TableGen
// AsmWriter bits get associated with the correct class.
def ARMAsmWriter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  bit isMCAsmWriter = 1;
}

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def ARM : Target {
  // Pull in Instruction Info:
  let InstructionSet = ARMInstrInfo;

  let AssemblyWriters = [ARMAsmWriter];
}
