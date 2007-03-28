//====- X86InstrMMX.td - Describe the X86 Instruction Set --*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
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
      : I<o, F, ops, asm, pattern>, TB, OpSize, Requires<[HasMMX]>;
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
def : Pat<(v1i64 (undef)), (IMPLICIT_DEF_VR64)>;

//===----------------------------------------------------------------------===//
// MMX Pattern Fragments
//===----------------------------------------------------------------------===//

def load_mmx : PatFrag<(ops node:$ptr), (v1i64 (load node:$ptr))>;

def bc_v8i8  : PatFrag<(ops node:$in), (v8i8  (bitconvert node:$in))>;
def bc_v4i16 : PatFrag<(ops node:$in), (v4i16 (bitconvert node:$in))>;
def bc_v2i32 : PatFrag<(ops node:$in), (v2i32 (bitconvert node:$in))>;
def bc_v1i64 : PatFrag<(ops node:$in), (v1i64 (bitconvert node:$in))>;

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
                                          (load_mmx addr:$src2)))))]>;
  }

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
                                   (bitconvert (load_mmx addr:$src2))))]>;
  }

  // MMXI_binop_rm_v1i64 - Simple MMX binary operator whose type is v1i64.
  //
  // FIXME: we could eliminate this and use MMXI_binop_rm instead if tblgen knew
  // to collapse (bitconvert VT to VT) into its operand.
  //
  multiclass MMXI_binop_rm_v1i64<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                 bit Commutable = 0> {
    def rr : MMXI<opc, MRMSrcReg, (ops VR64:$dst, VR64:$src1, VR64:$src2),
                  !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2}"),
                  [(set VR64:$dst, (v1i64 (OpNode VR64:$src1, VR64:$src2)))]> {
      let isCommutable = Commutable;
    }
    def rm : MMXI<opc, MRMSrcMem, (ops VR64:$dst, VR64:$src1, i64mem:$src2),
                  !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2}"),
                  [(set VR64:$dst,
                    (OpNode VR64:$src1,(load_mmx addr:$src2)))]>;
  }

  multiclass MMXI_binop_rmi_int<bits<8> opc, bits<8> opc2, Format ImmForm,
                                string OpcodeStr, Intrinsic IntId> {
    def rr : MMXI<opc, MRMSrcReg, (ops VR64:$dst, VR64:$src1, VR64:$src2),
                  !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2}"),
                  [(set VR64:$dst, (IntId VR64:$src1, VR64:$src2))]>;
    def rm : MMXI<opc, MRMSrcMem, (ops VR64:$dst, VR64:$src1, i64mem:$src2),
                  !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2}"),
                  [(set VR64:$dst, (IntId VR64:$src1,
                                    (bitconvert (load_mmx addr:$src2))))]>;
    def ri : MMXIi8<opc2, ImmForm, (ops VR64:$dst, VR64:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2}"),
                    [(set VR64:$dst, (IntId VR64:$src1,
                                      (scalar_to_vector (i32 imm:$src2))))]>;
  }
}

//===----------------------------------------------------------------------===//
// MMX EMMS Instruction
//===----------------------------------------------------------------------===//

def MMX_EMMS : MMXI<0x77, RawFrm, (ops), "emms", [(int_x86_mmx_emms)]>;

//===----------------------------------------------------------------------===//
// MMX Scalar Instructions
//===----------------------------------------------------------------------===//

// Arithmetic Instructions

// -- Addition
defm MMX_PADDB : MMXI_binop_rm<0xFC, "paddb", add, v8i8, 1>;
defm MMX_PADDW : MMXI_binop_rm<0xFD, "paddw", add, v4i16, 1>;
defm MMX_PADDD : MMXI_binop_rm<0xFE, "paddd", add, v2i32, 1>;

defm MMX_PADDSB  : MMXI_binop_rm_int<0xEC, "paddsb" , int_x86_mmx_padds_b, 1>;
defm MMX_PADDSW  : MMXI_binop_rm_int<0xED, "paddsw" , int_x86_mmx_padds_w, 1>;

defm MMX_PADDUSB : MMXI_binop_rm_int<0xDC, "paddusb", int_x86_mmx_paddus_b, 1>;
defm MMX_PADDUSW : MMXI_binop_rm_int<0xDD, "paddusw", int_x86_mmx_paddus_w, 1>;

// -- Subtraction
defm MMX_PSUBB : MMXI_binop_rm<0xF8, "psubb", sub, v8i8>;
defm MMX_PSUBW : MMXI_binop_rm<0xF9, "psubw", sub, v4i16>;
defm MMX_PSUBD : MMXI_binop_rm<0xFA, "psubd", sub, v2i32>;

defm MMX_PSUBSB  : MMXI_binop_rm_int<0xE8, "psubsb" , int_x86_mmx_psubs_b>;
defm MMX_PSUBSW  : MMXI_binop_rm_int<0xE9, "psubsw" , int_x86_mmx_psubs_w>;

defm MMX_PSUBUSB : MMXI_binop_rm_int<0xD8, "psubusb", int_x86_mmx_psubus_b>;
defm MMX_PSUBUSW : MMXI_binop_rm_int<0xD9, "psubusw", int_x86_mmx_psubus_w>;

// -- Multiplication
defm MMX_PMULLW  : MMXI_binop_rm<0xD5, "pmullw", mul, v4i16, 1>;
defm MMX_PMULHW  : MMXI_binop_rm_int<0xE5, "pmulhw" , int_x86_mmx_pmulh_w , 1>;

// -- Multiply and Add
defm MMX_PMADDWD : MMXI_binop_rm_int<0xF5, "pmaddwd", int_x86_mmx_pmadd_wd, 1>;

// Logical Instructions
defm MMX_PAND : MMXI_binop_rm_v1i64<0xDB, "pand", and, 1>;
defm MMX_POR  : MMXI_binop_rm_v1i64<0xEB, "por" , or,  1>;
defm MMX_PXOR : MMXI_binop_rm_v1i64<0xEF, "pxor", xor, 1>;

let isTwoAddress = 1 in {
  def MMX_PANDNrr : MMXI<0xDF, MRMSrcReg,
                         (ops VR64:$dst, VR64:$src1, VR64:$src2),
                         "pandn {$src2, $dst|$dst, $src2}",
                         [(set VR64:$dst, (v1i64 (and (vnot VR64:$src1),
                                                  VR64:$src2)))]>;
  def MMX_PANDNrm : MMXI<0xDF, MRMSrcMem,
                         (ops VR64:$dst, VR64:$src1, i64mem:$src2),
                         "pandn {$src2, $dst|$dst, $src2}",
                         [(set VR64:$dst, (v1i64 (and (vnot VR64:$src1),
                                                  (load addr:$src2))))]>;
}

// Shift Instructions
defm MMX_PSRLW : MMXI_binop_rmi_int<0xD1, 0x71, MRM2r, "psrlw",
                                    int_x86_mmx_psrl_w>;
defm MMX_PSRLD : MMXI_binop_rmi_int<0xD2, 0x72, MRM2r, "psrld",
                                    int_x86_mmx_psrl_d>;
defm MMX_PSRLQ : MMXI_binop_rmi_int<0xD3, 0x73, MRM2r, "psrlq",
                                    int_x86_mmx_psrl_q>;

defm MMX_PSLLW : MMXI_binop_rmi_int<0xF1, 0x71, MRM6r, "psllw",
                                    int_x86_mmx_psll_w>;
defm MMX_PSLLD : MMXI_binop_rmi_int<0xF2, 0x72, MRM6r, "pslld",
                                    int_x86_mmx_psll_d>;
defm MMX_PSLLQ : MMXI_binop_rmi_int<0xF3, 0x73, MRM6r, "psllq",
                                    int_x86_mmx_psll_q>;

defm MMX_PSRAW : MMXI_binop_rmi_int<0xE1, 0x71, MRM4r, "psraw",
                                    int_x86_mmx_psra_w>;
defm MMX_PSRAD : MMXI_binop_rmi_int<0xE2, 0x72, MRM4r, "psrad",
                                    int_x86_mmx_psra_d>;

// Comparison Instructions
defm MMX_PCMPEQB : MMXI_binop_rm_int<0x74, "pcmpeqb", int_x86_mmx_pcmpeq_b>;
defm MMX_PCMPEQW : MMXI_binop_rm_int<0x75, "pcmpeqw", int_x86_mmx_pcmpeq_w>;
defm MMX_PCMPEQD : MMXI_binop_rm_int<0x76, "pcmpeqd", int_x86_mmx_pcmpeq_d>;

defm MMX_PCMPGTB : MMXI_binop_rm_int<0x64, "pcmpgtb", int_x86_mmx_pcmpgt_b>;
defm MMX_PCMPGTW : MMXI_binop_rm_int<0x65, "pcmpgtw", int_x86_mmx_pcmpgt_w>;
defm MMX_PCMPGTD : MMXI_binop_rm_int<0x66, "pcmpgtd", int_x86_mmx_pcmpgt_d>;

// Conversion Instructions
def MMX_UNPCKH_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isUNPCKHMask(N);
}]>;

def MMX_UNPCKL_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isUNPCKLMask(N);
}]>;

// -- Unpack Instructions
let isTwoAddress = 1 in {
  // Unpack High Packed Data Instructions
  def MMX_PUNPCKHBWrr : MMXI<0x68, MRMSrcReg, 
                             (ops VR64:$dst, VR64:$src1, VR64:$src2),
                             "punpckhbw {$src2, $dst|$dst, $src2}",
                             [(set VR64:$dst,
                               (v8i8 (vector_shuffle VR64:$src1, VR64:$src2,
                                      MMX_UNPCKH_shuffle_mask)))]>;
  def MMX_PUNPCKHBWrm : MMXI<0x68, MRMSrcMem, 
                             (ops VR64:$dst, VR64:$src1, i64mem:$src2),
                             "punpckhbw {$src2, $dst|$dst, $src2}",
                             [(set VR64:$dst,
                               (v8i8 (vector_shuffle VR64:$src1,
                                      (bc_v8i8 (load_mmx addr:$src2)),
                                      MMX_UNPCKH_shuffle_mask)))]>;

  def MMX_PUNPCKHWDrr : MMXI<0x69, MRMSrcReg, 
                             (ops VR64:$dst, VR64:$src1, VR64:$src2),
                             "punpckhwd {$src2, $dst|$dst, $src2}",
                             [(set VR64:$dst,
                               (v4i16 (vector_shuffle VR64:$src1, VR64:$src2,
                                       MMX_UNPCKH_shuffle_mask)))]>;
  def MMX_PUNPCKHWDrm : MMXI<0x69, MRMSrcMem, 
                             (ops VR64:$dst, VR64:$src1, i64mem:$src2),
                             "punpckhwd {$src2, $dst|$dst, $src2}",
                             [(set VR64:$dst,
                               (v4i16 (vector_shuffle VR64:$src1,
                                       (bc_v4i16 (load_mmx addr:$src2)),
                                       MMX_UNPCKH_shuffle_mask)))]>;

  def MMX_PUNPCKHDQrr : MMXI<0x6A, MRMSrcReg, 
                             (ops VR64:$dst, VR64:$src1, VR64:$src2),
                             "punpckhdq {$src2, $dst|$dst, $src2}",
                             [(set VR64:$dst,
                               (v2i32 (vector_shuffle VR64:$src1, VR64:$src2,
                                       MMX_UNPCKH_shuffle_mask)))]>;
  def MMX_PUNPCKHDQrm : MMXI<0x6A, MRMSrcMem,
                             (ops VR64:$dst, VR64:$src1, i64mem:$src2),
                             "punpckhdq {$src2, $dst|$dst, $src2}",
                             [(set VR64:$dst,
                               (v2i32 (vector_shuffle VR64:$src1,
                                       (bc_v2i32 (load_mmx addr:$src2)),
                                       MMX_UNPCKH_shuffle_mask)))]>;

  // Unpack Low Packed Data Instructions
  def MMX_PUNPCKLBWrr : MMXI<0x60, MRMSrcReg,
                             (ops VR64:$dst, VR64:$src1, VR64:$src2),
                             "punpcklbw {$src2, $dst|$dst, $src2}",
                             [(set VR64:$dst,
                               (v8i8 (vector_shuffle VR64:$src1, VR64:$src2,
                                      MMX_UNPCKL_shuffle_mask)))]>;
  def MMX_PUNPCKLBWrm : MMXI<0x60, MRMSrcMem,
                             (ops VR64:$dst, VR64:$src1, i64mem:$src2),
                             "punpcklbw {$src2, $dst|$dst, $src2}",
                             [(set VR64:$dst,
                               (v8i8 (vector_shuffle VR64:$src1,
                                      (bc_v8i8 (load_mmx addr:$src2)),
                                      MMX_UNPCKL_shuffle_mask)))]>;

  def MMX_PUNPCKLWDrr : MMXI<0x61, MRMSrcReg,
                             (ops VR64:$dst, VR64:$src1, VR64:$src2),
                             "punpcklwd {$src2, $dst|$dst, $src2}",
                             [(set VR64:$dst,
                               (v4i16 (vector_shuffle VR64:$src1, VR64:$src2,
                                       MMX_UNPCKL_shuffle_mask)))]>;
  def MMX_PUNPCKLWDrm : MMXI<0x61, MRMSrcMem,
                             (ops VR64:$dst, VR64:$src1, i64mem:$src2),
                             "punpcklwd {$src2, $dst|$dst, $src2}",
                             [(set VR64:$dst,
                               (v4i16 (vector_shuffle VR64:$src1,
                                       (bc_v4i16 (load_mmx addr:$src2)),
                                       MMX_UNPCKL_shuffle_mask)))]>;

  def MMX_PUNPCKLDQrr : MMXI<0x62, MRMSrcReg, 
                             (ops VR64:$dst, VR64:$src1, VR64:$src2),
                             "punpckldq {$src2, $dst|$dst, $src2}",
                             [(set VR64:$dst,
                               (v2i32 (vector_shuffle VR64:$src1, VR64:$src2,
                                       MMX_UNPCKL_shuffle_mask)))]>;
  def MMX_PUNPCKLDQrm : MMXI<0x62, MRMSrcMem, 
                             (ops VR64:$dst, VR64:$src1, i64mem:$src2),
                             "punpckldq {$src2, $dst|$dst, $src2}",
                             [(set VR64:$dst,
                               (v2i32 (vector_shuffle VR64:$src1,
                                       (bc_v2i32 (load_mmx addr:$src2)),
                                       MMX_UNPCKL_shuffle_mask)))]>;
}

// -- Pack Instructions
defm MMX_PACKSSWB : MMXI_binop_rm_int<0x63, "packsswb", int_x86_mmx_packsswb>;
defm MMX_PACKSSDW : MMXI_binop_rm_int<0x6B, "packssdw", int_x86_mmx_packssdw>;
defm MMX_PACKUSWB : MMXI_binop_rm_int<0x67, "packuswb", int_x86_mmx_packuswb>;

// Data Transfer Instructions
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
                    [(set VR64:$dst, (load_mmx addr:$src))]>;
def MOVQ64mr : MMXI<0x7F, MRMDestMem, (ops i64mem:$dst, VR64:$src),
                    "movq {$src, $dst|$dst, $src}",
                    [(store (v1i64 VR64:$src), addr:$dst)]>;

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
                   Requires<[HasMMX]>;
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
// Alias Instructions
//===----------------------------------------------------------------------===//

// Alias instructions that map zero vector to pxor.
// FIXME: remove when we can teach regalloc that xor reg, reg is ok.
let isReMaterializable = 1 in {
  def MMX_V_SET0       : MMXI<0xEF, MRMInitReg, (ops VR64:$dst),
                              "pxor $dst, $dst",
                              [(set VR64:$dst, (v1i64 immAllZerosV))]>;
  def MMX_V_SETALLONES : MMXI<0x76, MRMInitReg, (ops VR64:$dst),
                              "pcmpeqd $dst, $dst",
                              [(set VR64:$dst, (v1i64 immAllOnesV))]>;
}

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// Store 64-bit integer vector values.
def : Pat<(store (v8i8  VR64:$src), addr:$dst),
          (MOVQ64mr addr:$dst, VR64:$src)>;
def : Pat<(store (v4i16 VR64:$src), addr:$dst),
          (MOVQ64mr addr:$dst, VR64:$src)>;
def : Pat<(store (v2i32 VR64:$src), addr:$dst),
          (MOVQ64mr addr:$dst, VR64:$src)>;

// 64-bit vector all zero's.
def : Pat<(v8i8  immAllZerosV), (MMX_V_SET0)>;
def : Pat<(v4i16 immAllZerosV), (MMX_V_SET0)>;
def : Pat<(v2i32 immAllZerosV), (MMX_V_SET0)>;
def : Pat<(v1i64 immAllZerosV), (MMX_V_SET0)>;

// 64-bit vector all one's.
def : Pat<(v8i8  immAllOnesV), (MMX_V_SETALLONES)>;
def : Pat<(v4i16 immAllOnesV), (MMX_V_SETALLONES)>;
def : Pat<(v2i32 immAllOnesV), (MMX_V_SETALLONES)>;
def : Pat<(v1i64 immAllOnesV), (MMX_V_SETALLONES)>;

// Bit convert.
def : Pat<(v8i8  (bitconvert (v1i64 VR64:$src))), (v8i8  VR64:$src)>;
def : Pat<(v8i8  (bitconvert (v2i32 VR64:$src))), (v8i8  VR64:$src)>;
def : Pat<(v8i8  (bitconvert (v4i16 VR64:$src))), (v8i8  VR64:$src)>;
def : Pat<(v4i16 (bitconvert (v1i64 VR64:$src))), (v4i16 VR64:$src)>;
def : Pat<(v4i16 (bitconvert (v2i32 VR64:$src))), (v4i16 VR64:$src)>;
def : Pat<(v4i16 (bitconvert (v8i8  VR64:$src))), (v4i16 VR64:$src)>;
def : Pat<(v2i32 (bitconvert (v1i64 VR64:$src))), (v2i32 VR64:$src)>;
def : Pat<(v2i32 (bitconvert (v4i16 VR64:$src))), (v2i32 VR64:$src)>;
def : Pat<(v2i32 (bitconvert (v8i8  VR64:$src))), (v2i32 VR64:$src)>;
def : Pat<(v1i64 (bitconvert (v2i32 VR64:$src))), (v1i64 VR64:$src)>;
def : Pat<(v1i64 (bitconvert (v4i16 VR64:$src))), (v1i64 VR64:$src)>;
def : Pat<(v1i64 (bitconvert (v8i8  VR64:$src))), (v1i64 VR64:$src)>;

// Splat v1i64
// MMX_SHUFFLE_get_shuf_imm xform function: convert vector_shuffle mask to
// MMX_PSHUF*, MMX_SHUFP* etc. imm.
def MMX_SHUFFLE_get_shuf_imm : SDNodeXForm<build_vector, [{
  return getI8Imm(X86::getShuffleSHUFImmediate(N));
}]>;

def MMX_splat_mask : PatLeaf<(build_vector), [{
  return X86::isSplatMask(N);
}], MMX_SHUFFLE_get_shuf_imm>;

let AddedComplexity = 10 in {
  def : Pat<(vector_shuffle (v1i64 VR64:$src), (undef),
             MMX_splat_mask:$sm),
            (MMX_PUNPCKHDQrr VR64:$src, VR64:$src)>;
  def : Pat<(vector_shuffle (v1i64 VR64:$src), (undef),
             MMX_UNPCKH_shuffle_mask:$sm),
            (MMX_PUNPCKHDQrr VR64:$src, VR64:$src)>;
}

def MMX_X86s2vec : SDNode<"X86ISD::S2VEC",  SDTypeProfile<1, 1, []>, []>;

// Scalar to v4i16 / v8i8. The source may be a GR32, but only the lower 8 or
// 16-bits matter.
def : Pat<(v8i8  (MMX_X86s2vec GR32:$src)), (MOVD64rr GR32:$src)>;
def : Pat<(v4i16 (MMX_X86s2vec GR32:$src)), (MOVD64rr GR32:$src)>;

// Some special case PANDN patterns.
def : Pat<(v1i64 (and (xor VR64:$src1, (bc_v1i64 (v2i32 immAllOnesV))),
                  VR64:$src2)),
          (MMX_PANDNrr VR64:$src1, VR64:$src2)>;
def : Pat<(v1i64 (and (xor VR64:$src1, (bc_v1i64 (v4i16 immAllOnesV))),
                  VR64:$src2)),
          (MMX_PANDNrr VR64:$src1, VR64:$src2)>;
def : Pat<(v1i64 (and (xor VR64:$src1, (bc_v1i64 (v8i8  immAllOnesV))),
                  VR64:$src2)),
          (MMX_PANDNrr VR64:$src1, VR64:$src2)>;

def : Pat<(v1i64 (and (xor VR64:$src1, (bc_v1i64 (v2i32 immAllOnesV))),
                  (load addr:$src2))),
          (MMX_PANDNrm VR64:$src1, addr:$src2)>;
def : Pat<(v1i64 (and (xor VR64:$src1, (bc_v1i64 (v4i16 immAllOnesV))),
                  (load addr:$src2))),
          (MMX_PANDNrm VR64:$src1, addr:$src2)>;
def : Pat<(v1i64 (and (xor VR64:$src1, (bc_v1i64 (v8i8  immAllOnesV))),
                  (load addr:$src2))),
          (MMX_PANDNrm VR64:$src1, addr:$src2)>;
