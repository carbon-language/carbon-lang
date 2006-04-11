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
// MMXi8 - MMX instructions with ImmT == Imm8 and TB prefix.
class MMXIi8<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : X86Inst<o, F, Imm8, ops, asm>, TB, Requires<[HasMMX]> {
  let Pattern = pattern;
}

// Some 'special' instructions
def IMPLICIT_DEF_VR64 : I<0, Pseudo, (ops VR64:$dst),
                          "#IMPLICIT_DEF $dst",
                          [(set VR64:$dst, (v8i8 (undef)))]>,
                        Requires<[HasMMX]>;

def : Pat<(v4i16 (undef)), (IMPLICIT_DEF_VR64)>,  Requires<[HasMMX]>;
def : Pat<(v2i32 (undef)), (IMPLICIT_DEF_VR64)>,  Requires<[HasMMX]>;

// Move Instructions
def MOVD64rr : I<0x6E, MRMSrcReg, (ops VR64:$dst, R32:$src),
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
def CVTTPS2PIrr: I<0x2C, MRMSrcReg, (ops VR64:$dst, VR128:$src),
                   "cvttps2pi {$src, $dst|$dst, $src}", []>, TB,
                 Requires<[HasSSE2]>;
def CVTTPS2PIrm: I<0x2C, MRMSrcMem, (ops VR64:$dst, f64mem:$src),
                   "cvttps2pi {$src, $dst|$dst, $src}", []>, TB,
                 Requires<[HasMMX]>;

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

