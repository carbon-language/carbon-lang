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
def SDT_MipsShilo : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;
def SDT_MipsDPA : SDTypeProfile<0, 2, [SDTCisVT<0, i32>, SDTCisSameAs<0, 1>]>;

class MipsDSPBase<string Opc, SDTypeProfile Prof> :
  SDNode<!strconcat("MipsISD::", Opc), Prof,
         [SDNPHasChain, SDNPInGlue, SDNPOutGlue]>;

class MipsDSPSideEffectBase<string Opc, SDTypeProfile Prof> :
  SDNode<!strconcat("MipsISD::", Opc), Prof,
         [SDNPHasChain, SDNPInGlue, SDNPOutGlue, SDNPSideEffect]>;

def MipsEXTP : MipsDSPSideEffectBase<"EXTP", SDT_MipsExtr>;
def MipsEXTPDP : MipsDSPSideEffectBase<"EXTPDP", SDT_MipsExtr>;
def MipsEXTR_S_H : MipsDSPSideEffectBase<"EXTR_S_H", SDT_MipsExtr>;
def MipsEXTR_W : MipsDSPSideEffectBase<"EXTR_W", SDT_MipsExtr>;
def MipsEXTR_R_W : MipsDSPSideEffectBase<"EXTR_R_W", SDT_MipsExtr>;
def MipsEXTR_RS_W : MipsDSPSideEffectBase<"EXTR_RS_W", SDT_MipsExtr>;

def MipsSHILO : MipsDSPBase<"SHILO", SDT_MipsShilo>;
def MipsMTHLIP : MipsDSPBase<"MTHLIP", SDT_MipsShilo>;

def MipsMULSAQ_S_W_PH : MipsDSPSideEffectBase<"MULSAQ_S_W_PH", SDT_MipsDPA>;
def MipsMAQ_S_W_PHL : MipsDSPSideEffectBase<"MAQ_S_W_PHL", SDT_MipsDPA>;
def MipsMAQ_S_W_PHR : MipsDSPSideEffectBase<"MAQ_S_W_PHR", SDT_MipsDPA>;
def MipsMAQ_SA_W_PHL : MipsDSPSideEffectBase<"MAQ_SA_W_PHL", SDT_MipsDPA>;
def MipsMAQ_SA_W_PHR : MipsDSPSideEffectBase<"MAQ_SA_W_PHR", SDT_MipsDPA>;

def MipsDPAU_H_QBL : MipsDSPBase<"DPAU_H_QBL", SDT_MipsDPA>;
def MipsDPAU_H_QBR : MipsDSPBase<"DPAU_H_QBR", SDT_MipsDPA>;
def MipsDPSU_H_QBL : MipsDSPBase<"DPSU_H_QBL", SDT_MipsDPA>;
def MipsDPSU_H_QBR : MipsDSPBase<"DPSU_H_QBR", SDT_MipsDPA>;
def MipsDPAQ_S_W_PH : MipsDSPSideEffectBase<"DPAQ_S_W_PH", SDT_MipsDPA>;
def MipsDPSQ_S_W_PH : MipsDSPSideEffectBase<"DPSQ_S_W_PH", SDT_MipsDPA>;
def MipsDPAQ_SA_L_W : MipsDSPSideEffectBase<"DPAQ_SA_L_W", SDT_MipsDPA>;
def MipsDPSQ_SA_L_W : MipsDSPSideEffectBase<"DPSQ_SA_L_W", SDT_MipsDPA>;

def MipsDPA_W_PH : MipsDSPBase<"DPA_W_PH", SDT_MipsDPA>;
def MipsDPS_W_PH : MipsDSPBase<"DPS_W_PH", SDT_MipsDPA>;
def MipsDPAQX_S_W_PH : MipsDSPSideEffectBase<"DPAQX_S_W_PH", SDT_MipsDPA>;
def MipsDPAQX_SA_W_PH : MipsDSPSideEffectBase<"DPAQX_SA_W_PH", SDT_MipsDPA>;
def MipsDPAX_W_PH : MipsDSPBase<"DPAX_W_PH", SDT_MipsDPA>;
def MipsDPSX_W_PH : MipsDSPBase<"DPSX_W_PH", SDT_MipsDPA>;
def MipsDPSQX_S_W_PH : MipsDSPSideEffectBase<"DPSQX_S_W_PH", SDT_MipsDPA>;
def MipsDPSQX_SA_W_PH : MipsDSPSideEffectBase<"DPSQX_SA_W_PH", SDT_MipsDPA>;
def MipsMULSA_W_PH : MipsDSPBase<"MULSA_W_PH", SDT_MipsDPA>;

def MipsMULT : MipsDSPBase<"MULT", SDT_MipsDPA>;
def MipsMULTU : MipsDSPBase<"MULTU", SDT_MipsDPA>;
def MipsMADD_DSP : MipsDSPBase<"MADD_DSP", SDT_MipsDPA>;
def MipsMADDU_DSP : MipsDSPBase<"MADDU_DSP", SDT_MipsDPA>;
def MipsMSUB_DSP : MipsDSPBase<"MSUB_DSP", SDT_MipsDPA>;
def MipsMSUBU_DSP : MipsDSPBase<"MSUBU_DSP", SDT_MipsDPA>;

// Flags.
class IsCommutable {
  bit isCommutable = 1;
}

class UseAC {
  list<Register> Uses = [AC0];
}

// Instruction encoding.
class MULSAQ_S_W_PH_ENC : DPA_W_PH_FMT<0b00110>;
class MAQ_S_W_PHL_ENC : DPA_W_PH_FMT<0b10100>;
class MAQ_S_W_PHR_ENC : DPA_W_PH_FMT<0b10110>;
class MAQ_SA_W_PHL_ENC : DPA_W_PH_FMT<0b10000>;
class MAQ_SA_W_PHR_ENC : DPA_W_PH_FMT<0b10010>;
class DPAU_H_QBL_ENC : DPA_W_PH_FMT<0b00011>;
class DPAU_H_QBR_ENC : DPA_W_PH_FMT<0b00111>;
class DPSU_H_QBL_ENC : DPA_W_PH_FMT<0b01011>;
class DPSU_H_QBR_ENC : DPA_W_PH_FMT<0b01111>;
class DPAQ_S_W_PH_ENC : DPA_W_PH_FMT<0b00100>;
class DPSQ_S_W_PH_ENC : DPA_W_PH_FMT<0b00101>;
class DPAQ_SA_L_W_ENC : DPA_W_PH_FMT<0b01100>;
class DPSQ_SA_L_W_ENC : DPA_W_PH_FMT<0b01101>;
class MULT_DSP_ENC : MULT_FMT<0b000000, 0b011000>;
class MULTU_DSP_ENC : MULT_FMT<0b000000, 0b011001>;
class MADD_DSP_ENC : MULT_FMT<0b011100, 0b000000>;
class MADDU_DSP_ENC : MULT_FMT<0b011100, 0b000001>;
class MSUB_DSP_ENC : MULT_FMT<0b011100, 0b000100>;
class MSUBU_DSP_ENC : MULT_FMT<0b011100, 0b000101>;
class BPOSGE32_ENC : BPOSGE32_FMT<0b11100>;

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
class SHILO_ENC : SHILO_R1_FMT<0b11010>;
class SHILOV_ENC : SHILO_R2_FMT<0b11011>;
class MTHLIP_ENC : SHILO_R2_FMT<0b11111>;

class DPA_W_PH_ENC : DPA_W_PH_FMT<0b00000>;
class DPS_W_PH_ENC : DPA_W_PH_FMT<0b00001>;
class DPAQX_S_W_PH_ENC : DPA_W_PH_FMT<0b11000>;
class DPAQX_SA_W_PH_ENC : DPA_W_PH_FMT<0b11010>;
class DPAX_W_PH_ENC : DPA_W_PH_FMT<0b01000>;
class DPSX_W_PH_ENC : DPA_W_PH_FMT<0b01001>;
class DPSQX_S_W_PH_ENC : DPA_W_PH_FMT<0b11001>;
class DPSQX_SA_W_PH_ENC : DPA_W_PH_FMT<0b11011>;
class MULSA_W_PH_ENC : DPA_W_PH_FMT<0b00010>;

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

class SHILO_R1_PSEUDO_BASE<SDPatternOperator OpNode, InstrItinClass itin,
                           Instruction realinst> :
  PseudoDSP<(outs), (ins simm16:$shift), [(OpNode immSExt6:$shift)]>,
  PseudoInstExpansion<(realinst AC0, simm16:$shift)> {
  list<Register> Defs = [DSPCtrl, AC0];
  list<Register> Uses = [AC0];
  InstrItinClass Itinerary = itin;
}

class SHILO_R1_DESC_BASE<string instr_asm> {
  dag OutOperandList = (outs ACRegs:$ac);
  dag InOperandList = (ins simm16:$shift);
  string AsmString = !strconcat(instr_asm, "\t$ac, $shift");
}

class SHILO_R2_PSEUDO_BASE<SDPatternOperator OpNode, InstrItinClass itin,
                           Instruction realinst> :
  PseudoDSP<(outs), (ins CPURegs:$rs), [(OpNode CPURegs:$rs)]>,
  PseudoInstExpansion<(realinst AC0, CPURegs:$rs)> {
  list<Register> Defs = [DSPCtrl, AC0];
  list<Register> Uses = [AC0];
  InstrItinClass Itinerary = itin;
}

class SHILO_R2_DESC_BASE<string instr_asm> {
  dag OutOperandList = (outs ACRegs:$ac);
  dag InOperandList = (ins CPURegs:$rs);
  string AsmString = !strconcat(instr_asm, "\t$ac, $rs");
}

class MTHLIP_DESC_BASE<string instr_asm> {
  dag OutOperandList = (outs ACRegs:$ac);
  dag InOperandList = (ins CPURegs:$rs);
  string AsmString = !strconcat(instr_asm, "\t$rs, $ac");
}

class DPA_W_PH_PSEUDO_BASE<SDPatternOperator OpNode, InstrItinClass itin,
                           Instruction realinst> :
  PseudoDSP<(outs), (ins CPURegs:$rs, CPURegs:$rt),
            [(OpNode CPURegs:$rs, CPURegs:$rt)]>,
  PseudoInstExpansion<(realinst AC0, CPURegs:$rs, CPURegs:$rt)> {
  list<Register> Defs = [DSPCtrl, AC0];
  list<Register> Uses = [AC0];
  InstrItinClass Itinerary = itin;
}

class DPA_W_PH_DESC_BASE<string instr_asm> {
  dag OutOperandList = (outs ACRegs:$ac);
  dag InOperandList = (ins CPURegs:$rs, CPURegs:$rt);
  string AsmString = !strconcat(instr_asm, "\t$ac, $rs, $rt");
}

class MULT_PSEUDO_BASE<SDPatternOperator OpNode, InstrItinClass itin,
                       Instruction realinst> :
  PseudoDSP<(outs), (ins CPURegs:$rs, CPURegs:$rt),
            [(OpNode CPURegs:$rs, CPURegs:$rt)]>,
  PseudoInstExpansion<(realinst AC0, CPURegs:$rs, CPURegs:$rt)> {
  list<Register> Defs = [DSPCtrl, AC0];
  InstrItinClass Itinerary = itin;
}

class MULT_DESC_BASE<string instr_asm> {
  dag OutOperandList = (outs ACRegs:$ac);
  dag InOperandList = (ins CPURegs:$rs, CPURegs:$rt);
  string AsmString = !strconcat(instr_asm, "\t$ac, $rs, $rt");
}

class BPOSGE32_PSEUDO_DESC_BASE<SDPatternOperator OpNode, InstrItinClass itin> :
  MipsPseudo<(outs CPURegs:$dst), (ins), "", [(set CPURegs:$dst, (OpNode))]> {
  list<Register> Uses = [DSPCtrl];
  bit usesCustomInserter = 1;
}

class BPOSGE32_DESC_BASE<string instr_asm, InstrItinClass itin> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins brtarget:$offset);
  string AsmString = !strconcat(instr_asm, "\t$offset");
  InstrItinClass Itinerary = itin;
  list<Register> Uses = [DSPCtrl];
  bit isBranch = 1;
  bit isTerminator = 1;
  bit hasDelaySlot = 1;
}

//===----------------------------------------------------------------------===//
// MIPS DSP Rev 1
//===----------------------------------------------------------------------===//

// Multiplication
class MULSAQ_S_W_PH_DESC : DPA_W_PH_DESC_BASE<"mulsaq_s.w.ph">;

class MAQ_S_W_PHL_DESC : DPA_W_PH_DESC_BASE<"maq_s.w.phl">;

class MAQ_S_W_PHR_DESC : DPA_W_PH_DESC_BASE<"maq_s.w.phr">;

class MAQ_SA_W_PHL_DESC : DPA_W_PH_DESC_BASE<"maq_sa.w.phl">;

class MAQ_SA_W_PHR_DESC : DPA_W_PH_DESC_BASE<"maq_sa.w.phr">;

// Dot product with accumulate/subtract
class DPAU_H_QBL_DESC : DPA_W_PH_DESC_BASE<"dpau.h.qbl">;

class DPAU_H_QBR_DESC : DPA_W_PH_DESC_BASE<"dpau.h.qbr">;

class DPSU_H_QBL_DESC : DPA_W_PH_DESC_BASE<"dpsu.h.qbl">;

class DPSU_H_QBR_DESC : DPA_W_PH_DESC_BASE<"dpsu.h.qbr">;

class DPAQ_S_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpaq_s.w.ph">;

class DPSQ_S_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpsq_s.w.ph">;

class DPAQ_SA_L_W_DESC : DPA_W_PH_DESC_BASE<"dpaq_sa.l.w">;

class DPSQ_SA_L_W_DESC : DPA_W_PH_DESC_BASE<"dpsq_sa.l.w">;

class MULT_DSP_DESC : MULT_DESC_BASE<"mult">;

class MULTU_DSP_DESC : MULT_DESC_BASE<"multu">;

class MADD_DSP_DESC : MULT_DESC_BASE<"madd">;

class MADDU_DSP_DESC : MULT_DESC_BASE<"maddu">;

class MSUB_DSP_DESC : MULT_DESC_BASE<"msub">;

class MSUBU_DSP_DESC : MULT_DESC_BASE<"msubu">;

// Misc
class BPOSGE32_DESC : BPOSGE32_DESC_BASE<"bposge32", NoItinerary>;

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

class SHILO_DESC : SHILO_R1_DESC_BASE<"shilo">;

class SHILOV_DESC : SHILO_R2_DESC_BASE<"shilov">;

class MTHLIP_DESC : MTHLIP_DESC_BASE<"mthlip">;

//===----------------------------------------------------------------------===//
// MIPS DSP Rev 2
// Dot product with accumulate/subtract
class DPA_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpa.w.ph">;

class DPS_W_PH_DESC : DPA_W_PH_DESC_BASE<"dps.w.ph">;

class DPAQX_S_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpaqx_s.w.ph">;

class DPAQX_SA_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpaqx_sa.w.ph">;

class DPAX_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpax.w.ph">;

class DPSX_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpsx.w.ph">;

class DPSQX_S_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpsqx_s.w.ph">;

class DPSQX_SA_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpsqx_sa.w.ph">;

class MULSA_W_PH_DESC : DPA_W_PH_DESC_BASE<"mulsa.w.ph">;

// Pseudos.
def BPOSGE32_PSEUDO : BPOSGE32_PSEUDO_DESC_BASE<int_mips_bposge32, NoItinerary>;

// Instruction defs.
// MIPS DSP Rev 1
def MULSAQ_S_W_PH : MULSAQ_S_W_PH_ENC, MULSAQ_S_W_PH_DESC;
def MAQ_S_W_PHL : MAQ_S_W_PHL_ENC, MAQ_S_W_PHL_DESC;
def MAQ_S_W_PHR : MAQ_S_W_PHR_ENC, MAQ_S_W_PHR_DESC;
def MAQ_SA_W_PHL : MAQ_SA_W_PHL_ENC, MAQ_SA_W_PHL_DESC;
def MAQ_SA_W_PHR : MAQ_SA_W_PHR_ENC, MAQ_SA_W_PHR_DESC;
def DPAU_H_QBL : DPAU_H_QBL_ENC, DPAU_H_QBL_DESC;
def DPAU_H_QBR : DPAU_H_QBR_ENC, DPAU_H_QBR_DESC;
def DPSU_H_QBL : DPSU_H_QBL_ENC, DPSU_H_QBL_DESC;
def DPSU_H_QBR : DPSU_H_QBR_ENC, DPSU_H_QBR_DESC;
def DPAQ_S_W_PH : DPAQ_S_W_PH_ENC, DPAQ_S_W_PH_DESC;
def DPSQ_S_W_PH : DPSQ_S_W_PH_ENC, DPSQ_S_W_PH_DESC;
def DPAQ_SA_L_W : DPAQ_SA_L_W_ENC, DPAQ_SA_L_W_DESC;
def DPSQ_SA_L_W : DPSQ_SA_L_W_ENC, DPSQ_SA_L_W_DESC;
def MULT_DSP : MULT_DSP_ENC, MULT_DSP_DESC;
def MULTU_DSP : MULTU_DSP_ENC, MULTU_DSP_DESC;
def MADD_DSP : MADD_DSP_ENC, MADD_DSP_DESC;
def MADDU_DSP : MADDU_DSP_ENC, MADDU_DSP_DESC;
def MSUB_DSP : MSUB_DSP_ENC, MSUB_DSP_DESC;
def MSUBU_DSP : MSUBU_DSP_ENC, MSUBU_DSP_DESC;
def BPOSGE32 : BPOSGE32_ENC, BPOSGE32_DESC;
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
def SHILO : SHILO_ENC, SHILO_DESC;
def SHILOV : SHILOV_ENC, SHILOV_DESC;
def MTHLIP : MTHLIP_ENC, MTHLIP_DESC;

// MIPS DSP Rev 2
let Predicates = [HasDSPR2] in {

def DPA_W_PH : DPA_W_PH_ENC, DPA_W_PH_DESC;
def DPS_W_PH : DPS_W_PH_ENC, DPS_W_PH_DESC;
def DPAQX_S_W_PH : DPAQX_S_W_PH_ENC, DPAQX_S_W_PH_DESC;
def DPAQX_SA_W_PH : DPAQX_SA_W_PH_ENC, DPAQX_SA_W_PH_DESC;
def DPAX_W_PH : DPAX_W_PH_ENC, DPAX_W_PH_DESC;
def DPSX_W_PH : DPSX_W_PH_ENC, DPSX_W_PH_DESC;
def DPSQX_S_W_PH : DPSQX_S_W_PH_ENC, DPSQX_S_W_PH_DESC;
def DPSQX_SA_W_PH : DPSQX_SA_W_PH_ENC, DPSQX_SA_W_PH_DESC;
def MULSA_W_PH : MULSA_W_PH_ENC, MULSA_W_PH_DESC;

}

// Pseudos.
def MULSAQ_S_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsMULSAQ_S_W_PH, NoItinerary,
                                                MULSAQ_S_W_PH>;
def MAQ_S_W_PHL_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsMAQ_S_W_PHL, NoItinerary,
                                              MAQ_S_W_PHL>;
def MAQ_S_W_PHR_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsMAQ_S_W_PHR, NoItinerary,
                                              MAQ_S_W_PHR>;
def MAQ_SA_W_PHL_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsMAQ_SA_W_PHL, NoItinerary,
                                               MAQ_SA_W_PHL>;
def MAQ_SA_W_PHR_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsMAQ_SA_W_PHR, NoItinerary,
                                               MAQ_SA_W_PHR>;
def DPAU_H_QBL_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPAU_H_QBL, NoItinerary,
                                             DPAU_H_QBL>;
def DPAU_H_QBR_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPAU_H_QBR, NoItinerary,
                                             DPAU_H_QBR>;
def DPSU_H_QBL_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPSU_H_QBL, NoItinerary,
                                             DPSU_H_QBL>;
def DPSU_H_QBR_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPSU_H_QBR, NoItinerary,
                                             DPSU_H_QBR>;
def DPAQ_S_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPAQ_S_W_PH, NoItinerary,
                                              DPAQ_S_W_PH>;
def DPSQ_S_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPSQ_S_W_PH, NoItinerary,
                                              DPSQ_S_W_PH>;
def DPAQ_SA_L_W_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPAQ_SA_L_W, NoItinerary,
                                              DPAQ_SA_L_W>;
def DPSQ_SA_L_W_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPSQ_SA_L_W, NoItinerary,
                                              DPSQ_SA_L_W>;

def MULT_DSP_PSEUDO : MULT_PSEUDO_BASE<MipsMULT, NoItinerary, MULT_DSP>,
                      IsCommutable;
def MULTU_DSP_PSEUDO : MULT_PSEUDO_BASE<MipsMULTU, NoItinerary, MULTU_DSP>,
                       IsCommutable;
def MADD_DSP_PSEUDO : MULT_PSEUDO_BASE<MipsMADD_DSP, NoItinerary, MADD_DSP>,
                      IsCommutable, UseAC;
def MADDU_DSP_PSEUDO : MULT_PSEUDO_BASE<MipsMADDU_DSP, NoItinerary, MADDU_DSP>,
                       IsCommutable, UseAC;
def MSUB_DSP_PSEUDO : MULT_PSEUDO_BASE<MipsMSUB_DSP, NoItinerary, MSUB_DSP>,
                      UseAC;
def MSUBU_DSP_PSEUDO : MULT_PSEUDO_BASE<MipsMSUBU_DSP, NoItinerary, MSUBU_DSP>,
                       UseAC;

def SHILO_PSEUDO : SHILO_R1_PSEUDO_BASE<MipsSHILO, NoItinerary, SHILO>;
def SHILOV_PSEUDO : SHILO_R2_PSEUDO_BASE<MipsSHILO, NoItinerary, SHILOV>;
def MTHLIP_PSEUDO : SHILO_R2_PSEUDO_BASE<MipsMTHLIP, NoItinerary, MTHLIP>;

let Predicates = [HasDSPR2] in {

def DPA_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPA_W_PH, NoItinerary, DPA_W_PH>;
def DPS_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPS_W_PH, NoItinerary, DPS_W_PH>;
def DPAQX_S_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPAQX_S_W_PH, NoItinerary,
                                               DPAQX_S_W_PH>;
def DPAQX_SA_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPAQX_SA_W_PH, NoItinerary,
                                                DPAQX_SA_W_PH>;
def DPAX_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPAX_W_PH, NoItinerary,
                                            DPAX_W_PH>;
def DPSX_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPSX_W_PH, NoItinerary,
                                            DPSX_W_PH>;
def DPSQX_S_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPSQX_S_W_PH, NoItinerary,
                                               DPSQX_S_W_PH>;
def DPSQX_SA_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPSQX_SA_W_PH, NoItinerary,
                                                DPSQX_SA_W_PH>;
def MULSA_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsMULSA_W_PH, NoItinerary,
                                             MULSA_W_PH>;

}

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
