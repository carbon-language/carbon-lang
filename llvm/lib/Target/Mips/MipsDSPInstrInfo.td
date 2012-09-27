//===- MipsDSPInstrInfo.td - DSP ASE instructions -*- tablegen ------------*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips DSP ASE instructions.
//
//===----------------------------------------------------------------------===//

// ImmLeaf
def immZExt2 : ImmLeaf<i32, [{return isUInt<2>(Imm);}]>;
def immZExt3 : ImmLeaf<i32, [{return isUInt<3>(Imm);}]>;
def immZExt4 : ImmLeaf<i32, [{return isUInt<4>(Imm);}]>;
def immZExt8 : ImmLeaf<i32, [{return isUInt<8>(Imm);}]>;
def immZExt10 : ImmLeaf<i32, [{return isUInt<10>(Imm);}]>;
def immSExt6 : ImmLeaf<i32, [{return isInt<6>(Imm);}]>;

// Mips-specific dsp nodes
def SDT_MipsExtr : SDTypeProfile<1, 1, [SDTCisVT<0, i32>, SDTCisSameAs<0, 1>]>;

class MipsDSPSideEffectBase<string Opc, SDTypeProfile Prof> :
  SDNode<!strconcat("MipsISD::", Opc), Prof,
         [SDNPHasChain, SDNPInGlue, SDNPOutGlue, SDNPSideEffect]>;

def MipsEXTP : MipsDSPSideEffectBase<"EXTP", SDT_MipsExtr>;
def MipsEXTPDP : MipsDSPSideEffectBase<"EXTPDP", SDT_MipsExtr>;
def MipsEXTR_S_H : MipsDSPSideEffectBase<"EXTR_S_H", SDT_MipsExtr>;
def MipsEXTR_W : MipsDSPSideEffectBase<"EXTR_W", SDT_MipsExtr>;
def MipsEXTR_R_W : MipsDSPSideEffectBase<"EXTR_R_W", SDT_MipsExtr>;
def MipsEXTR_RS_W : MipsDSPSideEffectBase<"EXTR_RS_W", SDT_MipsExtr>;

// Instruction encoding.
class EXTP_ENC : EXTR_W_TY1_FMT<0b00010>;
class EXTPV_ENC : EXTR_W_TY1_FMT<0b00011>;
class EXTPDP_ENC : EXTR_W_TY1_FMT<0b01010>;
class EXTPDPV_ENC : EXTR_W_TY1_FMT<0b01011>;
class EXTR_W_ENC : EXTR_W_TY1_FMT<0b00000>;
class EXTRV_W_ENC : EXTR_W_TY1_FMT<0b00001>;
class EXTR_R_W_ENC : EXTR_W_TY1_FMT<0b00100>;
class EXTRV_R_W_ENC : EXTR_W_TY1_FMT<0b00101>;
class EXTR_RS_W_ENC : EXTR_W_TY1_FMT<0b00110>;
class EXTRV_RS_W_ENC : EXTR_W_TY1_FMT<0b00111>;
class EXTR_S_H_ENC : EXTR_W_TY1_FMT<0b01110>;
class EXTRV_S_H_ENC : EXTR_W_TY1_FMT<0b01111>;

// Instruction desc.
class EXTR_W_TY1_R2_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                              InstrItinClass itin> {
  dag OutOperandList = (outs CPURegs:$rt);
  dag InOperandList = (ins ACRegs:$ac, CPURegs:$shift_rs);
  string AsmString = !strconcat(instr_asm, "\t$rt, $ac, $shift_rs");
  InstrItinClass Itinerary = itin;
  list<Register> Defs = [DSPCtrl];
}

class EXTR_W_TY1_R1_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                              InstrItinClass itin> {
  dag OutOperandList = (outs CPURegs:$rt);
  dag InOperandList = (ins ACRegs:$ac, uimm16:$shift_rs);
  string AsmString = !strconcat(instr_asm, "\t$rt, $ac, $shift_rs");
  InstrItinClass Itinerary = itin;
  list<Register> Defs = [DSPCtrl];
}

//===----------------------------------------------------------------------===//
// MIPS DSP Rev 1
//===----------------------------------------------------------------------===//

// Extr
class EXTP_DESC : EXTR_W_TY1_R1_DESC_BASE<"extp", MipsEXTP, NoItinerary>;

class EXTPV_DESC : EXTR_W_TY1_R2_DESC_BASE<"extpv", MipsEXTP, NoItinerary>;

class EXTPDP_DESC : EXTR_W_TY1_R1_DESC_BASE<"extpdp", MipsEXTPDP, NoItinerary>;

class EXTPDPV_DESC : EXTR_W_TY1_R2_DESC_BASE<"extpdpv", MipsEXTPDP,
                                             NoItinerary>;

class EXTR_W_DESC : EXTR_W_TY1_R1_DESC_BASE<"extr.w", MipsEXTR_W, NoItinerary>;

class EXTRV_W_DESC : EXTR_W_TY1_R2_DESC_BASE<"extrv.w", MipsEXTR_W,
                                             NoItinerary>;

class EXTR_R_W_DESC : EXTR_W_TY1_R1_DESC_BASE<"extr_r.w", MipsEXTR_R_W,
                                              NoItinerary>;

class EXTRV_R_W_DESC : EXTR_W_TY1_R2_DESC_BASE<"extrv_r.w", MipsEXTR_R_W,
                                               NoItinerary>;

class EXTR_RS_W_DESC : EXTR_W_TY1_R1_DESC_BASE<"extr_rs.w", MipsEXTR_RS_W,
                                               NoItinerary>;

class EXTRV_RS_W_DESC : EXTR_W_TY1_R2_DESC_BASE<"extrv_rs.w", MipsEXTR_RS_W,
                                                NoItinerary>;

class EXTR_S_H_DESC : EXTR_W_TY1_R1_DESC_BASE<"extr_s.h", MipsEXTR_S_H,
                                              NoItinerary>;

class EXTRV_S_H_DESC : EXTR_W_TY1_R2_DESC_BASE<"extrv_s.h", MipsEXTR_S_H,
                                               NoItinerary>;

// Instruction defs.
// MIPS DSP Rev 1
def EXTP : EXTP_ENC, EXTP_DESC;
def EXTPV : EXTPV_ENC, EXTPV_DESC;
def EXTPDP : EXTPDP_ENC, EXTPDP_DESC;
def EXTPDPV : EXTPDPV_ENC, EXTPDPV_DESC;
def EXTR_W : EXTR_W_ENC, EXTR_W_DESC;
def EXTRV_W : EXTRV_W_ENC, EXTRV_W_DESC;
def EXTR_R_W : EXTR_R_W_ENC, EXTR_R_W_DESC;
def EXTRV_R_W : EXTRV_R_W_ENC, EXTRV_R_W_DESC;
def EXTR_RS_W : EXTR_RS_W_ENC, EXTR_RS_W_DESC;
def EXTRV_RS_W : EXTRV_RS_W_ENC, EXTRV_RS_W_DESC;
def EXTR_S_H : EXTR_S_H_ENC, EXTR_S_H_DESC;
def EXTRV_S_H : EXTRV_S_H_ENC, EXTRV_S_H_DESC;

// Patterns.
class DSPPat<dag pattern, dag result, Predicate pred = HasDSP> :
  Pat<pattern, result>, Requires<[pred]>;

class BitconvertPat<ValueType DstVT, ValueType SrcVT, RegisterClass DstRC,
                    RegisterClass SrcRC> :
   DSPPat<(DstVT (bitconvert (SrcVT SrcRC:$src))),
          (COPY_TO_REGCLASS SrcRC:$src, DstRC)>;

def : BitconvertPat<i32, v2i16, CPURegs, DSPRegs>;
def : BitconvertPat<i32, v4i8, CPURegs, DSPRegs>;
def : BitconvertPat<v2i16, i32, DSPRegs, CPURegs>;
def : BitconvertPat<v4i8, i32, DSPRegs, CPURegs>;

def : DSPPat<(v2i16 (load addr:$a)),
             (v2i16 (COPY_TO_REGCLASS (LW addr:$a), DSPRegs))>;
def : DSPPat<(v4i8 (load addr:$a)),
             (v4i8 (COPY_TO_REGCLASS (LW addr:$a), DSPRegs))>;
def : DSPPat<(store (v2i16 DSPRegs:$val), addr:$a),
             (SW (COPY_TO_REGCLASS DSPRegs:$val, CPURegs), addr:$a)>;
def : DSPPat<(store (v4i8 DSPRegs:$val), addr:$a),
             (SW (COPY_TO_REGCLASS DSPRegs:$val, CPURegs), addr:$a)>;

// Extr patterns.
class EXTR_W_TY1_R2_Pat<SDPatternOperator OpNode, Instruction Instr> :
  DSPPat<(i32 (OpNode CPURegs:$rs)), (Instr AC0, CPURegs:$rs)>;

class EXTR_W_TY1_R1_Pat<SDPatternOperator OpNode, Instruction Instr> :
  DSPPat<(i32 (OpNode immZExt5:$shift)), (Instr AC0, immZExt5:$shift)>;

def : EXTR_W_TY1_R1_Pat<MipsEXTP, EXTP>;
def : EXTR_W_TY1_R2_Pat<MipsEXTP, EXTPV>;
def : EXTR_W_TY1_R1_Pat<MipsEXTPDP, EXTPDP>;
def : EXTR_W_TY1_R2_Pat<MipsEXTPDP, EXTPDPV>;
def : EXTR_W_TY1_R1_Pat<MipsEXTR_W, EXTR_W>;
def : EXTR_W_TY1_R2_Pat<MipsEXTR_W, EXTRV_W>;
def : EXTR_W_TY1_R1_Pat<MipsEXTR_R_W, EXTR_R_W>;
def : EXTR_W_TY1_R2_Pat<MipsEXTR_R_W, EXTRV_R_W>;
def : EXTR_W_TY1_R1_Pat<MipsEXTR_RS_W, EXTR_RS_W>;
def : EXTR_W_TY1_R2_Pat<MipsEXTR_RS_W, EXTRV_RS_W>;
def : EXTR_W_TY1_R1_Pat<MipsEXTR_S_H, EXTR_S_H>;
def : EXTR_W_TY1_R2_Pat<MipsEXTR_S_H, EXTRV_S_H>;
