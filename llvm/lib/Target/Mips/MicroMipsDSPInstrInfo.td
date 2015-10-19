//===- MicroMipsDSPInstrInfo.td - Micromips DSP instructions -*- tablegen *-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes MicroMips DSP instructions.
//
//===----------------------------------------------------------------------===//

// Instruction encoding.
class ADDQ_PH_MM_ENC : POOL32A_3R_FMT<"addq.ph", 0b00000001101>;
class ADDQ_S_PH_MM_ENC : POOL32A_3R_FMT<"addq_s.ph", 0b10000001101>;
class ADDQ_S_W_MM_ENC : POOL32A_3RB0_FMT<"addq_s.w", 0b1100000101>;
class ADDQH_PH_MMR2_ENC : POOL32A_3R_FMT<"addqh.ph", 0b00001001101>;
class ADDQH_R_PH_MMR2_ENC : POOL32A_3R_FMT<"addqh_r.ph", 0b10001001101>;
class ADDQH_W_MMR2_ENC: POOL32A_3R_FMT<"addqh.w", 0b00010001101>;
class ADDQH_R_W_MMR2_ENC : POOL32A_3R_FMT<"addqh_r.w", 0b10010001101>;
class ADDU_PH_MMR2_ENC : POOL32A_3R_FMT<"addu.ph", 0b00100001101>;
class ADDU_S_PH_MMR2_ENC : POOL32A_3R_FMT<"addu_s.ph", 0b10100001101>;
class ADDU_QB_MM_ENC : POOL32A_3R_FMT<"addu.qb", 0b00011001101>;
class ADDU_S_QB_MM_ENC : POOL32A_3R_FMT<"addu_s.qb", 0b10011001101>;
class ADDUH_QB_MMR2_ENC : POOL32A_3R_FMT<"adduh.qb", 0b00101001101>;
class ADDUH_R_QB_MMR2_ENC : POOL32A_3R_FMT<"adduh_r.qb", 0b10101001101>;
class ADDSC_MM_ENC : POOL32A_3RB0_FMT<"addsc", 0b1110000101>;
class ADDWC_MM_ENC : POOL32A_3RB0_FMT<"addwc", 0b1111000101>;
class DPA_W_PH_MMR2_ENC : POOL32A_AC2R_FMT<"dpa.w.ph", 0b00000010>;
class DPAQ_S_W_PH_MM_ENC : POOL32A_AC2R_FMT<"dpaq_s.w.ph", 0b00001010>;
class DPAQ_SA_L_W_MM_ENC : POOL32A_AC2R_FMT<"dpaq_sa.l.w", 0b01001010>;
class DPAQX_S_W_PH_MMR2_ENC : POOL32A_AC2R_FMT<"dpaqx_s.w.ph", 0b10001010>;
class DPAQX_SA_W_PH_MMR2_ENC : POOL32A_AC2R_FMT<"dpaqx_sa.w.ph", 0b11001010>;
class DPAU_H_QBL_MM_ENC : POOL32A_AC2R_FMT<"dpau.h.qbl", 0b10000010>;
class DPAU_H_QBR_MM_ENC : POOL32A_AC2R_FMT<"dpau.h.qbr", 0b11000010>;
class DPAX_W_PH_MMR2_ENC : POOL32A_AC2R_FMT<"dpax.w.ph", 0b01000010>;
class ABSQ_S_PH_MM_ENC : POOL32A_2R_FMT<"absq_s.ph", 0b0001000100>;
class ABSQ_S_W_MM_ENC : POOL32A_2R_FMT<"absq_s.w", 0b0010000100>;
class ABSQ_S_QB_MMR2_ENC : POOL32A_2R_FMT<"absq_s.qb", 0b0000000100>;
class INSV_MM_ENC : POOL32A_2R_FMT<"insv", 0b0100000100>;
class MADD_DSP_MM_ENC : POOL32A_2RAC_FMT<"madd", 0b00101010>;
class MADDU_DSP_MM_ENC : POOL32A_2RAC_FMT<"maddu", 0b01101010>;
class MSUB_DSP_MM_ENC : POOL32A_2RAC_FMT<"msub", 0b10101010>;
class MSUBU_DSP_MM_ENC : POOL32A_2RAC_FMT<"msubu", 0b11101010>;
class MULT_DSP_MM_ENC : POOL32A_2RAC_FMT<"mult", 0b00110010>;
class MULTU_DSP_MM_ENC : POOL32A_2RAC_FMT<"multu", 0b01110010>;

// Instruction defs.
class ABSQ_S_PH_MM_R2_DESC_BASE<string opstr, SDPatternOperator OpNode,
                                InstrItinClass itin, RegisterOperand ROD,
                                RegisterOperand ROS = ROD> {
  dag OutOperandList = (outs ROD:$rt);
  dag InOperandList = (ins ROS:$rs);
  string AsmString = !strconcat(opstr, "\t$rt, $rs");
  list<dag> Pattern = [(set ROD:$rt, (OpNode ROS:$rs))];
  InstrItinClass Itinerary = itin;
}
class ABSQ_S_PH_MM_DESC : ABSQ_S_PH_MM_R2_DESC_BASE<
  "absq_s.ph", int_mips_absq_s_ph, NoItinerary, DSPROpnd>, Defs<[DSPOutFlag20]>;
class ABSQ_S_W_MM_DESC : ABSQ_S_PH_MM_R2_DESC_BASE<
  "absq_s.w", int_mips_absq_s_w, NoItinerary, GPR32Opnd>, Defs<[DSPOutFlag20]>;
class ABSQ_S_QB_MMR2_DESC : ABSQ_S_PH_MM_R2_DESC_BASE<
  "absq_s.qb", int_mips_absq_s_qb, NoItinerary, DSPROpnd>, Defs<[DSPOutFlag20]>;

// microMIPS DSP Rev 1
def ADDQ_PH_MM : DspMMRel, ADDQ_PH_MM_ENC, ADDQ_PH_DESC;
def ADDQ_S_PH_MM : DspMMRel, ADDQ_S_PH_MM_ENC, ADDQ_S_PH_DESC;
def ADDQ_S_W_MM : DspMMRel, ADDQ_S_W_MM_ENC, ADDQ_S_W_DESC;
def ADDU_QB_MM : DspMMRel, ADDU_QB_MM_ENC, ADDU_QB_DESC;
def ADDU_S_QB_MM : DspMMRel, ADDU_S_QB_MM_ENC, ADDU_S_QB_DESC;
def ADDSC_MM : DspMMRel, ADDSC_MM_ENC, ADDSC_DESC;
def ADDWC_MM : DspMMRel, ADDWC_MM_ENC, ADDWC_DESC;
def DPAQ_S_W_PH_MM : DspMMRel, DPAQ_S_W_PH_MM_ENC, DPAQ_S_W_PH_DESC;
def DPAQ_SA_L_W_MM : DspMMRel, DPAQ_SA_L_W_MM_ENC, DPAQ_SA_L_W_DESC;
def DPAU_H_QBL_MM : DspMMRel, DPAU_H_QBL_MM_ENC, DPAU_H_QBL_DESC;
def DPAU_H_QBR_MM : DspMMRel, DPAU_H_QBR_MM_ENC, DPAU_H_QBR_DESC;
def ABSQ_S_PH_MM : DspMMRel, ABSQ_S_PH_MM_ENC, ABSQ_S_PH_MM_DESC;
def ABSQ_S_W_MM : DspMMRel, ABSQ_S_W_MM_ENC, ABSQ_S_W_MM_DESC;
def INSV_MM : DspMMRel, INSV_MM_ENC, INSV_DESC;
def MADD_DSP_MM : DspMMRel, MADD_DSP_MM_ENC, MADD_DSP_DESC;
def MADDU_DSP_MM : DspMMRel, MADDU_DSP_MM_ENC, MADDU_DSP_DESC;
def MSUB_DSP_MM : DspMMRel, MSUB_DSP_MM_ENC, MSUB_DSP_DESC;
def MSUBU_DSP_MM : DspMMRel, MSUBU_DSP_MM_ENC, MSUBU_DSP_DESC;
def MULT_DSP_MM : DspMMRel, MULT_DSP_MM_ENC, MULT_DSP_DESC;
def MULTU_DSP_MM : DspMMRel, MULTU_DSP_MM_ENC, MULTU_DSP_DESC;
// microMIPS DSP Rev 2
def ABSQ_S_QB_MMR2 : DspMMRel, ABSQ_S_QB_MMR2_ENC, ABSQ_S_QB_MMR2_DESC,
                     ISA_DSPR2;
def ADDQH_PH_MMR2 : DspMMRel, ADDQH_PH_MMR2_ENC, ADDQH_PH_DESC, ISA_DSPR2;
def ADDQH_R_PH_MMR2 : DspMMRel, ADDQH_R_PH_MMR2_ENC, ADDQH_R_PH_DESC, ISA_DSPR2;
def ADDQH_W_MMR2 : DspMMRel, ADDQH_W_MMR2_ENC, ADDQH_W_DESC, ISA_DSPR2;
def ADDQH_R_W_MMR2 : DspMMRel, ADDQH_R_W_MMR2_ENC, ADDQH_R_W_DESC, ISA_DSPR2;
def ADDU_PH_MMR2 : DspMMRel, ADDU_PH_MMR2_ENC, ADDU_PH_DESC, ISA_DSPR2;
def ADDU_S_PH_MMR2 : DspMMRel, ADDU_S_PH_MMR2_ENC, ADDU_S_PH_DESC, ISA_DSPR2;
def ADDUH_QB_MMR2 : DspMMRel, ADDUH_QB_MMR2_ENC, ADDUH_QB_DESC, ISA_DSPR2;
def ADDUH_R_QB_MMR2 : DspMMRel, ADDUH_R_QB_MMR2_ENC, ADDUH_R_QB_DESC, ISA_DSPR2;
def DPA_W_PH_MMR2 : DspMMRel, DPA_W_PH_MMR2_ENC, DPA_W_PH_DESC, ISA_DSPR2;
def DPAQX_S_W_PH_MMR2 : DspMMRel, DPAQX_S_W_PH_MMR2_ENC, DPAQX_S_W_PH_DESC,
                        ISA_DSPR2;
def DPAQX_SA_W_PH_MMR2 : DspMMRel, DPAQX_SA_W_PH_MMR2_ENC, DPAQX_SA_W_PH_DESC,
                         ISA_DSPR2;
def DPAX_W_PH_MMR2 : DspMMRel, DPAX_W_PH_MMR2_ENC, DPAX_W_PH_DESC, ISA_DSPR2;
