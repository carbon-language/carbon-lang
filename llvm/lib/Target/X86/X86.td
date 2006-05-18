//===- X86.td - Target definition file for the Intel X86 arch ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This is a target description file for the Intel i386 architecture, refered to
// here as the "X86" architecture.
//
//===----------------------------------------------------------------------===//

// Get the target-independent interfaces which we are implementing...
//
include "../Target.td"

//===----------------------------------------------------------------------===//
// X86 Subtarget features.
//
 
def Feature64Bit     : SubtargetFeature<"64bit", "Is64Bit", "true",
                                        "Enable 64-bit instructions">;
def FeatureMMX       : SubtargetFeature<"mmx","X86SSELevel", "MMX",
                                        "Enable MMX instructions">;
def FeatureSSE1      : SubtargetFeature<"sse", "X86SSELevel", "SSE1",
                                        "Enable SSE instructions">;
def FeatureSSE2      : SubtargetFeature<"sse2", "X86SSELevel", "SSE2",
                                        "Enable SSE2 instructions">;
def FeatureSSE3      : SubtargetFeature<"sse3", "X86SSELevel", "SSE3",
                                        "Enable SSE3 instructions">;
def Feature3DNow     : SubtargetFeature<"3dnow", "X863DNowLevel", "ThreeDNow",
                                        "Enable 3DNow! instructions">;
def Feature3DNowA    : SubtargetFeature<"3dnowa", "X863DNowLevel", "ThreeDNowA",
                                        "Enable 3DNow! Athlon instructions">;

//===----------------------------------------------------------------------===//
// X86 processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

def : Proc<"generic",         []>;
def : Proc<"i386",            []>;
def : Proc<"i486",            []>;
def : Proc<"pentium",         []>;
def : Proc<"pentium-mmx",     [FeatureMMX]>;
def : Proc<"i686",            []>;
def : Proc<"pentiumpro",      []>;
def : Proc<"pentium2",        [FeatureMMX]>;
def : Proc<"pentium3",        [FeatureMMX, FeatureSSE1]>;
def : Proc<"pentium-m",       [FeatureMMX, FeatureSSE1, FeatureSSE2]>;
def : Proc<"pentium4",        [FeatureMMX, FeatureSSE1, FeatureSSE2]>;
def : Proc<"x86-64",          [FeatureMMX, FeatureSSE1, FeatureSSE2,
                               Feature64Bit]>;
def : Proc<"yonah",           [FeatureMMX, FeatureSSE1, FeatureSSE2,
                               FeatureSSE3]>;
def : Proc<"prescott",        [FeatureMMX, FeatureSSE1, FeatureSSE2,
                               FeatureSSE3]>;
def : Proc<"nocona",          [FeatureMMX, FeatureSSE1, FeatureSSE2,
                               FeatureSSE3, Feature64Bit]>;

def : Proc<"k6",              [FeatureMMX]>;
def : Proc<"k6-2",            [FeatureMMX, Feature3DNow]>;
def : Proc<"k6-3",            [FeatureMMX, Feature3DNow]>;
def : Proc<"athlon",          [FeatureMMX, Feature3DNow, Feature3DNowA]>;
def : Proc<"athlon-tbird",    [FeatureMMX, Feature3DNow, Feature3DNowA]>;
def : Proc<"athlon-4",        [FeatureMMX, FeatureSSE1, Feature3DNow,
                               Feature3DNowA]>;
def : Proc<"athlon-xp",       [FeatureMMX, FeatureSSE1, Feature3DNow,
                               Feature3DNowA]>;
def : Proc<"athlon-mp",       [FeatureMMX, FeatureSSE1, Feature3DNow,
                               Feature3DNowA]>;
def : Proc<"k8",              [FeatureMMX, FeatureSSE1, FeatureSSE2,
                               Feature3DNow, Feature3DNowA, Feature64Bit]>;
def : Proc<"opteron",         [FeatureMMX, FeatureSSE1, FeatureSSE2,
                               Feature3DNow, Feature3DNowA, Feature64Bit]>;
def : Proc<"athlon64",        [FeatureMMX, FeatureSSE1, FeatureSSE2,
                               Feature3DNow, Feature3DNowA, Feature64Bit]>;
def : Proc<"athlon-fx",       [FeatureMMX, FeatureSSE1, FeatureSSE2,
                               Feature3DNow, Feature3DNowA, Feature64Bit]>;

def : Proc<"winchip-c6",      [FeatureMMX]>;
def : Proc<"winchip2",        [FeatureMMX, Feature3DNow]>;
def : Proc<"c3",              [FeatureMMX, Feature3DNow]>;
def : Proc<"c3-2",            [FeatureMMX, FeatureSSE1]>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "X86RegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "X86InstrInfo.td"

def X86InstrInfo : InstrInfo {

  // Define how we want to layout our TargetSpecific information field... This
  // should be kept up-to-date with the fields in the X86InstrInfo.h file.
  let TSFlagsFields = ["FormBits",
                       "hasOpSizePrefix",
                       "Prefix",
                       "ImmTypeBits",
                       "FPFormBits",
                       "Opcode"];
  let TSFlagsShifts = [0,
                       6,
                       7,
                       11,
                       13,
                       16];
}

// The X86 target supports two different syntaxes for emitting machine code.
// This is controlled by the -x86-asm-syntax={att|intel}
def ATTAsmWriter : AsmWriter {
  string AsmWriterClassName  = "ATTAsmPrinter";
  int Variant = 0;
}
def IntelAsmWriter : AsmWriter {
  string AsmWriterClassName  = "IntelAsmPrinter";
  int Variant = 1;
}


def X86 : Target {
  // Information about the instructions...
  let InstructionSet = X86InstrInfo;

  let AssemblyWriters = [ATTAsmWriter, IntelAsmWriter];
}
