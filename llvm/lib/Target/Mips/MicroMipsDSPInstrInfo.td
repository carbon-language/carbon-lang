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
class ADDU_QB_MM_ENC : POOL32A_3R_FMT<"addu.qb", 0b00011001101>;
class DPA_W_PH_MMR2_ENC : POOL32A_AC2R_FMT<"dpa.w.ph", 0b00000010>;
class DPAQ_S_W_PH_MM_ENC : POOL32A_AC2R_FMT<"dpaq_s.w.ph", 0b00001010>;
class DPAQ_SA_L_W_MM_ENC : POOL32A_AC2R_FMT<"dpaq_sa.l.w", 0b01001010>;
class DPAQX_S_W_PH_MMR2_ENC : POOL32A_AC2R_FMT<"dpaqx_s.w.ph", 0b10001010>;
class DPAQX_SA_W_PH_MMR2_ENC : POOL32A_AC2R_FMT<"dpaqx_sa.w.ph", 0b11001010>;
class DPAU_H_QBL_MM_ENC : POOL32A_AC2R_FMT<"dpau.h.qbl", 0b10000010>;
class DPAU_H_QBR_MM_ENC : POOL32A_AC2R_FMT<"dpau.h.qbr", 0b11000010>;
class DPAX_W_PH_MMR2_ENC : POOL32A_AC2R_FMT<"dpax.w.ph", 0b01000010>;

// Instruction defs.
// microMIPS DSP Rev 1
def ADDU_QB_MM : DspMMRel, ADDU_QB_MM_ENC, ADDU_QB_DESC;
def DPAQ_S_W_PH_MM : DspMMRel, DPAQ_S_W_PH_MM_ENC, DPAQ_S_W_PH_DESC;
def DPAQ_SA_L_W_MM : DspMMRel, DPAQ_SA_L_W_MM_ENC, DPAQ_SA_L_W_DESC;
def DPAU_H_QBL_MM : DspMMRel, DPAU_H_QBL_MM_ENC, DPAU_H_QBL_DESC;
def DPAU_H_QBR_MM : DspMMRel, DPAU_H_QBR_MM_ENC, DPAU_H_QBR_DESC;
// microMIPS DSP Rev 2
def DPA_W_PH_MMR2 : DspMMRel, DPA_W_PH_MMR2_ENC, DPA_W_PH_DESC, ISA_DSPR2;
def DPAQX_S_W_PH_MMR2 : DspMMRel, DPAQX_S_W_PH_MMR2_ENC, DPAQX_S_W_PH_DESC,
                        ISA_DSPR2;
def DPAQX_SA_W_PH_MMR2 : DspMMRel, DPAQX_SA_W_PH_MMR2_ENC, DPAQX_SA_W_PH_DESC,
                         ISA_DSPR2;
def DPAX_W_PH_MMR2 : DspMMRel, DPAX_W_PH_MMR2_ENC, DPAX_W_PH_DESC, ISA_DSPR2;
