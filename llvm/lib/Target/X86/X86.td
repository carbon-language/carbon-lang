//===- X86.td - Target definition file for the Intel X86 ---*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This is a target description file for the Intel i386 architecture, refered to
// here as the "X86" architecture.
//
//===----------------------------------------------------------------------===//

// Get the target-independent interfaces which we are implementing...
//
include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// X86 Subtarget features.
//===----------------------------------------------------------------------===//

def FeatureCMOV    : SubtargetFeature<"cmov","HasCMov", "true",
                                      "Enable conditional move instructions">;


def FeatureMMX     : SubtargetFeature<"mmx","X86SSELevel", "MMX",
                                      "Enable MMX instructions">;
def FeatureSSE1    : SubtargetFeature<"sse", "X86SSELevel", "SSE1",
                                      "Enable SSE instructions",
                                      // SSE codegen depends on cmovs, and all
                                      // SSE1+ processors support them. 
                                      [FeatureMMX, FeatureCMOV]>;
def FeatureSSE2    : SubtargetFeature<"sse2", "X86SSELevel", "SSE2",
                                      "Enable SSE2 instructions",
                                      [FeatureSSE1]>;
def FeatureSSE3    : SubtargetFeature<"sse3", "X86SSELevel", "SSE3",
                                      "Enable SSE3 instructions",
                                      [FeatureSSE2]>;
def FeatureSSSE3   : SubtargetFeature<"ssse3", "X86SSELevel", "SSSE3",
                                      "Enable SSSE3 instructions",
                                      [FeatureSSE3]>;
def FeatureSSE41   : SubtargetFeature<"sse41", "X86SSELevel", "SSE41",
                                      "Enable SSE 4.1 instructions",
                                      [FeatureSSSE3]>;
def FeatureSSE42   : SubtargetFeature<"sse42", "X86SSELevel", "SSE42",
                                      "Enable SSE 4.2 instructions",
                                      [FeatureSSE41]>;
def Feature3DNow   : SubtargetFeature<"3dnow", "X863DNowLevel", "ThreeDNow",
                                      "Enable 3DNow! instructions">;
def Feature3DNowA  : SubtargetFeature<"3dnowa", "X863DNowLevel", "ThreeDNowA",
                                      "Enable 3DNow! Athlon instructions",
                                      [Feature3DNow]>;
// All x86-64 hardware has SSE2, but we don't mark SSE2 as an implied
// feature, because SSE2 can be disabled (e.g. for compiling OS kernels)
// without disabling 64-bit mode.
def Feature64Bit   : SubtargetFeature<"64bit", "HasX86_64", "true",
                                      "Support 64-bit instructions",
                                      [FeatureCMOV]>;
def FeatureSlowBTMem : SubtargetFeature<"slow-bt-mem", "IsBTMemSlow", "true",
                                       "Bit testing of memory is slow">;
def FeatureFastUAMem : SubtargetFeature<"fast-unaligned-mem",
                                        "IsUAMemFast", "true",
                                        "Fast unaligned memory access">;
def FeatureSSE4A   : SubtargetFeature<"sse4a", "HasSSE4A", "true",
                                      "Support SSE 4a instructions">;

def FeatureAVX     : SubtargetFeature<"avx", "HasAVX", "true",
                                      "Enable AVX instructions">;
def FeatureCLMUL   : SubtargetFeature<"clmul", "HasCLMUL", "true",
                               "Enable carry-less multiplication instructions">;
def FeatureFMA3    : SubtargetFeature<"fma3", "HasFMA3", "true",
                                     "Enable three-operand fused multiple-add">;
def FeatureFMA4    : SubtargetFeature<"fma4", "HasFMA4", "true",
                                      "Enable four-operand fused multiple-add">;
def FeatureVectorUAMem : SubtargetFeature<"vector-unaligned-mem",
                                          "HasVectorUAMem", "true",
                 "Allow unaligned memory operands on vector/SIMD instructions">;
def FeatureAES     : SubtargetFeature<"aes", "HasAES", "true",
                                      "Enable AES instructions">;

//===----------------------------------------------------------------------===//
// X86 processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

def : Proc<"generic",         []>;
def : Proc<"i386",            []>;
def : Proc<"i486",            []>;
def : Proc<"i586",            []>;
def : Proc<"pentium",         []>;
def : Proc<"pentium-mmx",     [FeatureMMX]>;
def : Proc<"i686",            []>;
def : Proc<"pentiumpro",      [FeatureCMOV]>;
def : Proc<"pentium2",        [FeatureMMX, FeatureCMOV]>;
def : Proc<"pentium3",        [FeatureSSE1]>;
def : Proc<"pentium-m",       [FeatureSSE2, FeatureSlowBTMem]>;
def : Proc<"pentium4",        [FeatureSSE2]>;
def : Proc<"x86-64",          [FeatureSSE2,   Feature64Bit, FeatureSlowBTMem]>;
def : Proc<"yonah",           [FeatureSSE3, FeatureSlowBTMem]>;
def : Proc<"prescott",        [FeatureSSE3, FeatureSlowBTMem]>;
def : Proc<"nocona",          [FeatureSSE3,   Feature64Bit, FeatureSlowBTMem]>;
def : Proc<"core2",           [FeatureSSSE3,  Feature64Bit, FeatureSlowBTMem]>;
def : Proc<"penryn",          [FeatureSSE41,  Feature64Bit, FeatureSlowBTMem]>;
def : Proc<"atom",            [FeatureSSE3,   Feature64Bit, FeatureSlowBTMem]>;
// "Arrandale" along with corei3 and corei5
def : Proc<"corei7",          [FeatureSSE42,  Feature64Bit, FeatureSlowBTMem,
                               FeatureFastUAMem, FeatureAES]>;
def : Proc<"nehalem",         [FeatureSSE42,  Feature64Bit, FeatureSlowBTMem,
                               FeatureFastUAMem]>;
// Westmere is a similar machine to nehalem with some additional features.
// Westmere is the corei3/i5/i7 path from nehalem to sandybridge
def : Proc<"westmere",         [FeatureSSE42,  Feature64Bit, FeatureSlowBTMem,
                                FeatureFastUAMem, FeatureAES]>;
// Sandy Bridge does not have FMA
// FIXME: Wikipedia says it does... it should have AES as well.
def : Proc<"sandybridge",     [FeatureSSE42,  FeatureAVX,   Feature64Bit]>;

def : Proc<"k6",              [FeatureMMX]>;
def : Proc<"k6-2",            [FeatureMMX,    Feature3DNow]>;
def : Proc<"k6-3",            [FeatureMMX,    Feature3DNow]>;
def : Proc<"athlon",          [FeatureMMX,    Feature3DNowA, FeatureSlowBTMem]>;
def : Proc<"athlon-tbird",    [FeatureMMX,    Feature3DNowA, FeatureSlowBTMem]>;
def : Proc<"athlon-4",        [FeatureSSE1,   Feature3DNowA, FeatureSlowBTMem]>;
def : Proc<"athlon-xp",       [FeatureSSE1,   Feature3DNowA, FeatureSlowBTMem]>;
def : Proc<"athlon-mp",       [FeatureSSE1,   Feature3DNowA, FeatureSlowBTMem]>;
def : Proc<"k8",              [FeatureSSE2,   Feature3DNowA, Feature64Bit,
                               FeatureSlowBTMem]>;
def : Proc<"opteron",         [FeatureSSE2,   Feature3DNowA, Feature64Bit,
                               FeatureSlowBTMem]>;
def : Proc<"athlon64",        [FeatureSSE2,   Feature3DNowA, Feature64Bit,
                               FeatureSlowBTMem]>;
def : Proc<"athlon-fx",       [FeatureSSE2,   Feature3DNowA, Feature64Bit,
                               FeatureSlowBTMem]>;
def : Proc<"k8-sse3",         [FeatureSSE3,   Feature3DNowA, Feature64Bit,
                               FeatureSlowBTMem]>;
def : Proc<"opteron-sse3",    [FeatureSSE3,   Feature3DNowA, Feature64Bit,
                               FeatureSlowBTMem]>;
def : Proc<"athlon64-sse3",   [FeatureSSE3,   Feature3DNowA, Feature64Bit,
                               FeatureSlowBTMem]>;
def : Proc<"amdfam10",        [FeatureSSE3,   FeatureSSE4A,
                               Feature3DNowA, Feature64Bit, FeatureSlowBTMem]>;
def : Proc<"barcelona",       [FeatureSSE3,   FeatureSSE4A,
                               Feature3DNowA, Feature64Bit, FeatureSlowBTMem]>;
def : Proc<"istanbul",        [Feature3DNowA, Feature64Bit, FeatureSSE4A,
                               Feature3DNowA]>;
def : Proc<"shanghai",        [Feature3DNowA, Feature64Bit, FeatureSSE4A,
                               Feature3DNowA]>;

def : Proc<"winchip-c6",      [FeatureMMX]>;
def : Proc<"winchip2",        [FeatureMMX, Feature3DNow]>;
def : Proc<"c3",              [FeatureMMX, Feature3DNow]>;
def : Proc<"c3-2",            [FeatureSSE1]>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "X86RegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "X86InstrInfo.td"

def X86InstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Calling Conventions
//===----------------------------------------------------------------------===//

include "X86CallingConv.td"


//===----------------------------------------------------------------------===//
// Assembly Parser
//===----------------------------------------------------------------------===//

// Currently the X86 assembly parser only supports ATT syntax.
def ATTAsmParser : AsmParser {
  string AsmParserClassName = "ATTAsmParser";
  int Variant = 0;

  // Discard comments in assembly strings.
  string CommentDelimiter = "#";

  // Recognize hard coded registers.
  string RegisterPrefix = "%";
}

//===----------------------------------------------------------------------===//
// Assembly Printers
//===----------------------------------------------------------------------===//

// The X86 target supports two different syntaxes for emitting machine code.
// This is controlled by the -x86-asm-syntax={att|intel}
def ATTAsmWriter : AsmWriter {
  string AsmWriterClassName  = "ATTInstPrinter";
  int Variant = 0;
  bit isMCAsmWriter = 1;
}
def IntelAsmWriter : AsmWriter {
  string AsmWriterClassName  = "IntelInstPrinter";
  int Variant = 1;
  bit isMCAsmWriter = 1;
}

def X86 : Target {
  // Information about the instructions...
  let InstructionSet = X86InstrInfo;

  let AssemblyParsers = [ATTAsmParser];

  let AssemblyWriters = [ATTAsmWriter, IntelAsmWriter];
}
