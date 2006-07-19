//====- X86InstrMMX.td - Describe the X86 Instruction Set -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86 MMX instruction set, defining the instructions,
// and properties of the instructions which are needed for code generation,
// machine code emission, and analysis.
//
//===----------------------------------------------------------------------===//

// Instruction templates
// MMXI   - MMX instructions with TB prefix.
// MMX2I  - MMX / SSE2 instructions with TB and OpSize prefixes.
// MMXIi8 - MMX instructions with ImmT == Imm8 and TB prefix.
class MMXI<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : I<o, F, ops, asm, pattern>, TB, Requires<[HasMMX]>;
class MMX2I<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : I<o, F, ops, asm, pattern>, TB, OpSize, Requires<[HasSSE2]>;
class MMXIi8<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : Ii8<o, F, ops, asm, pattern>, TB, Requires<[HasMMX]>;

// Some 'special' instructions
def IMPLICIT_DEF_VR64 : I<0, Pseudo, (ops VR64:$dst),
                          "#IMPLICIT_DEF $dst",
                          [(set VR64:$dst, (v8i8 (undef)))]>,
                        Requires<[HasMMX]>;

def : Pat<(v4i16 (undef)), (IMPLICIT_DEF_VR64)>,  Requires<[HasMMX]>;
def : Pat<(v2i32 (undef)), (IMPLICIT_DEF_VR64)>,  Requires<[HasMMX]>;

// Move Instructions
def MOVD64rr : I<0x6E, MRMSrcReg, (ops VR64:$dst, GR32:$src),
                 "movd {$src, $dst|$dst, $src}", []>, TB,
               Requires<[HasMMX]>;
def MOVD64rm : I<0x6E, MRMSrcMem, (ops VR64:$dst, i32mem:$src),
                 "movd {$src, $dst|$dst, $src}", []>, TB,
               Requires<[HasMMX]>;
def MOVD64mr : I<0x7E, MRMDestMem, (ops i32mem:$dst, VR64:$src),
                 "movd {$src, $dst|$dst, $src}", []>, TB,
               Requires<[HasMMX]>;

def MOVQ64rr : I<0x6F, MRMSrcReg, (ops VR64:$dst, VR64:$src),
                 "movq {$src, $dst|$dst, $src}", []>, TB,
               Requires<[HasMMX]>;
def MOVQ64rm : I<0x6F, MRMSrcMem, (ops VR64:$dst, i64mem:$src),
                 "movq {$src, $dst|$dst, $src}", []>, TB,
               Requires<[HasMMX]>;
def MOVQ64mr : I<0x7F, MRMDestMem, (ops i64mem:$dst, VR64:$src),
                 "movq {$src, $dst|$dst, $src}", []>, TB,
               Requires<[HasMMX]>;

// Conversion instructions
def CVTPI2PSrr : MMXI<0x2A, MRMSrcReg, (ops VR128:$dst, VR64:$src),
                      "cvtpi2ps {$src, $dst|$dst, $src}", []>;
def CVTPI2PSrm : MMXI<0x2A, MRMSrcMem, (ops VR128:$dst, i64mem:$src),
                      "cvtpi2ps {$src, $dst|$dst, $src}", []>;
def CVTPI2PDrr : MMX2I<0x2A, MRMSrcReg, (ops VR128:$dst, VR64:$src),
                       "cvtpi2pd {$src, $dst|$dst, $src}", []>;
def CVTPI2PDrm : MMX2I<0x2A, MRMSrcMem, (ops VR128:$dst, i64mem:$src),
                       "cvtpi2pd {$src, $dst|$dst, $src}", []>;
def CVTTPS2PIrr: I<0x2C, MRMSrcReg, (ops VR64:$dst, VR128:$src),
                   "cvttps2pi {$src, $dst|$dst, $src}", []>, TB,
                 Requires<[HasSSE2]>;
def CVTTPS2PIrm: I<0x2C, MRMSrcMem, (ops VR64:$dst, f64mem:$src),
                   "cvttps2pi {$src, $dst|$dst, $src}", []>, TB,
                 Requires<[HasMMX]>;
def CVTPS2PIrr : MMXI<0x2D, MRMSrcReg, (ops VR64:$dst, VR128:$src),
                      "cvtps2pi {$src, $dst|$dst, $src}", []>;
def CVTPS2PIrm : MMXI<0x2D, MRMSrcMem, (ops VR64:$dst, f64mem:$src),
                      "cvtps2pi {$src, $dst|$dst, $src}", []>;
def CVTPD2PIrr : MMX2I<0x2D, MRMSrcReg, (ops VR64:$dst, VR128:$src),
                       "cvtpd2pi {$src, $dst|$dst, $src}", []>;
def CVTPD2PIrm : MMX2I<0x2D, MRMSrcMem, (ops VR64:$dst, f128mem:$src),
                       "cvtpd2pi {$src, $dst|$dst, $src}", []>;

// Shuffle and unpack instructions
def PSHUFWri : MMXIi8<0x70, MRMSrcReg,
                      (ops VR64:$dst, VR64:$src1, i8imm:$src2),
                      "pshufw {$src2, $src1, $dst|$dst, $src1, $src2}", []>;
def PSHUFWmi : MMXIi8<0x70, MRMSrcMem,
                      (ops VR64:$dst, i64mem:$src1, i8imm:$src2),
                      "pshufw {$src2, $src1, $dst|$dst, $src1, $src2}", []>;

// Misc.
def MOVNTQ   : I<0xE7, MRMDestMem, (ops i64mem:$dst, VR64:$src),
                 "movntq {$src, $dst|$dst, $src}", []>, TB,
               Requires<[HasMMX]>;

def MASKMOVQ : I<0xF7, MRMDestMem, (ops VR64:$src, VR64:$mask),
                 "maskmovq {$mask, $src|$src, $mask}", []>, TB,
               Requires<[HasMMX]>;

