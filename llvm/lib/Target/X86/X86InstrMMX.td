//====- X86InstrMMX.td - Describe the X86 Instruction Set --*- tablegen -*-===//
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

//===----------------------------------------------------------------------===//
// Instruction templates
//===----------------------------------------------------------------------===//

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

// 64-bit vector undef's.
def : Pat<(v8i8  (undef)), (IMPLICIT_DEF_VR64)>;
def : Pat<(v4i16 (undef)), (IMPLICIT_DEF_VR64)>;
def : Pat<(v2i32 (undef)), (IMPLICIT_DEF_VR64)>;

//===----------------------------------------------------------------------===//
// MMX Pattern Fragments
//===----------------------------------------------------------------------===//

def loadv2i32 : PatFrag<(ops node:$ptr), (v2i32 (load node:$ptr))>;

//===----------------------------------------------------------------------===//
// MMX Multiclasses
//===----------------------------------------------------------------------===//

let isTwoAddress = 1 in {
  // MMXI_binop_rm - Simple MMX binary operator.
  multiclass MMXI_binop_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           ValueType OpVT, bit Commutable = 0> {
    def rr : MMXI<opc, MRMSrcReg, (ops VR64:$dst, VR64:$src1, VR64:$src2),
                  !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2}"),
                  [(set VR64:$dst, (OpVT (OpNode VR64:$src1, VR64:$src2)))]> {
      let isCommutable = Commutable;
    }
    def rm : MMXI<opc, MRMSrcMem, (ops VR64:$dst, VR64:$src1, i64mem:$src2),
                  !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2}"),
                  [(set VR64:$dst, (OpVT (OpNode VR64:$src1,
                                         (bitconvert
                                          (loadv2i32 addr:$src2)))))]>;
  }
}

let isTwoAddress = 1 in {
  multiclass MMXI_binop_rm_int<bits<8> opc, string OpcodeStr, Intrinsic IntId,
                               bit Commutable = 0> {
    def rr : MMXI<opc, MRMSrcReg, (ops VR64:$dst, VR64:$src1, VR64:$src2),
                 !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2}"),
                 [(set VR64:$dst, (IntId VR64:$src1, VR64:$src2))]> {
      let isCommutable = Commutable;
    }
    def rm : MMXI<opc, MRMSrcMem, (ops VR64:$dst, VR64:$src1, i64mem:$src2),
                 !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2}"),
                 [(set VR64:$dst, (IntId VR64:$src1,
                                   (bitconvert (loadv2i32 addr:$src2))))]>;
  }
}

//===----------------------------------------------------------------------===//
// MMX EMMS Instruction
//===----------------------------------------------------------------------===//

def EMMS : MMXI<0x77, RawFrm, (ops), "emms", [(int_x86_mmx_emms)]>;

//===----------------------------------------------------------------------===//
// MMX Scalar Instructions
//===----------------------------------------------------------------------===//

// Arithmetic Instructions
defm MMX_PADDB : MMXI_binop_rm<0xFC, "paddb", add, v8i8, 1>;
defm MMX_PADDW : MMXI_binop_rm<0xFD, "paddw", add, v4i16, 1>;
defm MMX_PADDD : MMXI_binop_rm<0xFE, "paddd", add, v2i32, 1>;

defm MMX_PADDSB  : MMXI_binop_rm_int<0xEC, "paddsb" , int_x86_mmx_padds_b, 1>;
defm MMX_PADDSW  : MMXI_binop_rm_int<0xED, "paddsw" , int_x86_mmx_padds_w, 1>;

defm MMX_PADDUSB : MMXI_binop_rm_int<0xDC, "paddusb", int_x86_mmx_paddus_b, 1>;
defm MMX_PADDUSW : MMXI_binop_rm_int<0xDD, "paddusw", int_x86_mmx_paddus_w, 1>;

defm MMX_PSUBB : MMXI_binop_rm<0xF8, "psubb", sub, v8i8>;
defm MMX_PSUBW : MMXI_binop_rm<0xF9, "psubw", sub, v4i16>;
defm MMX_PSUBD : MMXI_binop_rm<0xFA, "psubd", sub, v2i32>;

defm MMX_PSUBSB  : MMXI_binop_rm_int<0xE8, "psubsb" , int_x86_mmx_psubs_b>;
defm MMX_PSUBSW  : MMXI_binop_rm_int<0xE9, "psubsw" , int_x86_mmx_psubs_w>;

defm MMX_PSUBUSB : MMXI_binop_rm_int<0xD8, "psubusb", int_x86_mmx_psubus_b>;
defm MMX_PSUBUSW : MMXI_binop_rm_int<0xD9, "psubusw", int_x86_mmx_psubus_w>;

// Move Instructions
def MOVD64rr : MMXI<0x6E, MRMSrcReg, (ops VR64:$dst, GR32:$src),
                    "movd {$src, $dst|$dst, $src}", []>;
def MOVD64rm : MMXI<0x6E, MRMSrcMem, (ops VR64:$dst, i32mem:$src),
                    "movd {$src, $dst|$dst, $src}", []>;
def MOVD64mr : MMXI<0x7E, MRMDestMem, (ops i32mem:$dst, VR64:$src),
                    "movd {$src, $dst|$dst, $src}", []>;

def MOVQ64rr : MMXI<0x6F, MRMSrcReg, (ops VR64:$dst, VR64:$src),
                    "movq {$src, $dst|$dst, $src}", []>;
def MOVQ64rm : MMXI<0x6F, MRMSrcMem, (ops VR64:$dst, i64mem:$src),
                    "movq {$src, $dst|$dst, $src}",
                    [(set VR64:$dst, (loadv2i32 addr:$src))]>;
def MOVQ64mr : MMXI<0x7F, MRMDestMem, (ops i64mem:$dst, VR64:$src),
                    "movq {$src, $dst|$dst, $src}",
                    [(store (v2i32 VR64:$src), addr:$dst)]>;

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

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// Store 64-bit integer vector values.
def : Pat<(store (v8i8  VR64:$src), addr:$dst),
          (MOVQ64mr addr:$dst, VR64:$src)>;
def : Pat<(store (v4i16 VR64:$src), addr:$dst),
          (MOVQ64mr addr:$dst, VR64:$src)>;

// Bit convert.
def : Pat<(v8i8  (bitconvert (v2i32 VR64:$src))), (v8i8  VR64:$src)>;
def : Pat<(v8i8  (bitconvert (v4i16 VR64:$src))), (v8i8  VR64:$src)>;
def : Pat<(v4i16 (bitconvert (v2i32 VR64:$src))), (v4i16 VR64:$src)>;
def : Pat<(v4i16 (bitconvert (v8i8  VR64:$src))), (v4i16 VR64:$src)>;
def : Pat<(v2i32 (bitconvert (v4i16 VR64:$src))), (v2i32 VR64:$src)>;
def : Pat<(v2i32 (bitconvert (v8i8  VR64:$src))), (v2i32 VR64:$src)>;
