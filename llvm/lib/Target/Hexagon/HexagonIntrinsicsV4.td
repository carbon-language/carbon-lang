//===- HexagonIntrinsicsV4.td - V4 Instruction intrinsics --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This is populated based on the following specs:
// Hexagon V4 Architecture Extensions
// Application-Level Specification
// 80-V9418-12 Rev. A
// June 15, 2010

// Polynomial multiply words
// Rdd=pmpyw(Rs,Rt)
def : T_RR_pat <M4_pmpyw, int_hexagon_M4_pmpyw>;
// Rxx^=pmpyw(Rs,Rt)
def : T_PRR_pat <M4_pmpyw_acc, int_hexagon_M4_pmpyw_acc>;

//Rxx^=asr(Rss,Rt)
def : T_PPR_pat <S2_asr_r_p_xor, int_hexagon_S2_asr_r_p_xor>;
//Rxx^=asl(Rss,Rt)
def : T_PPR_pat <S2_asl_r_p_xor, int_hexagon_S2_asl_r_p_xor>;
//Rxx^=lsr(Rss,Rt)
def : T_PPR_pat <S2_lsr_r_p_xor, int_hexagon_S2_lsr_r_p_xor>;
//Rxx^=lsl(Rss,Rt)
def : T_PPR_pat <S2_lsl_r_p_xor, int_hexagon_S2_lsl_r_p_xor>;

// Multiply and use upper result
def : MType_R32_pat <int_hexagon_M2_mpysu_up, M2_mpysu_up>;
def : MType_R32_pat <int_hexagon_M2_mpy_up_s1, M2_mpy_up_s1>;
def : MType_R32_pat <int_hexagon_M2_hmmpyh_s1, M2_hmmpyh_s1>;
def : MType_R32_pat <int_hexagon_M2_hmmpyl_s1, M2_hmmpyl_s1>;
def : MType_R32_pat <int_hexagon_M2_mpy_up_s1_sat, M2_mpy_up_s1_sat>;

def : T_P_pat <S2_brevp, int_hexagon_S2_brevp>;

def: T_P_pat  <S2_ct0p,      int_hexagon_S2_ct0p>;
def: T_P_pat  <S2_ct1p,      int_hexagon_S2_ct1p>;
def: T_RR_pat<C4_nbitsset,  int_hexagon_C4_nbitsset>;
def: T_RR_pat<C4_nbitsclr,  int_hexagon_C4_nbitsclr>;
def: T_RI_pat<C4_nbitsclri, int_hexagon_C4_nbitsclri>;

def : T_RP_pat <A4_boundscheck, int_hexagon_A4_boundscheck>;

def : T_PR_pat<A4_tlbmatch, int_hexagon_A4_tlbmatch>;

def : Pat <(int_hexagon_M4_mpyrr_addr IntRegs:$src1, IntRegs:$src2,
                                      IntRegs:$src3),
           (M4_mpyrr_addr IntRegs:$src1, IntRegs:$src2, IntRegs:$src3)>;

def : T_IRR_pat <M4_mpyrr_addi, int_hexagon_M4_mpyrr_addi>;
def : T_IRI_pat <M4_mpyri_addi, int_hexagon_M4_mpyri_addi>;
def : T_RIR_pat <M4_mpyri_addr_u2, int_hexagon_M4_mpyri_addr_u2>;
def : T_RRI_pat <M4_mpyri_addr, int_hexagon_M4_mpyri_addr>;
// Multiply 32x32 and use upper result
def : T_RRR_pat <M4_mac_up_s1_sat, int_hexagon_M4_mac_up_s1_sat>;
def : T_RRR_pat <M4_nac_up_s1_sat, int_hexagon_M4_nac_up_s1_sat>;

def : T_PP_pat<A4_andnp, int_hexagon_A4_andnp>;
def : T_PP_pat<A4_ornp,  int_hexagon_A4_ornp>;

// Extract bitfield
def : T_PP_pat  <S4_extractp_rp, int_hexagon_S4_extractp_rp>;
def : T_RP_pat  <S4_extract_rp, int_hexagon_S4_extract_rp>;
def : T_PII_pat <S4_extractp, int_hexagon_S4_extractp>;
def : T_RII_pat <S4_extract, int_hexagon_S4_extract>;

// Shift an immediate left by register amount
def : T_IR_pat<S4_lsli, int_hexagon_S4_lsli>;

// Logical xor with xor accumulation
def : T_PPP_pat<M4_xor_xacc, int_hexagon_M4_xor_xacc>;

// Shift and add/sub/and/or
def : T_IRI_pat <S4_andi_asl_ri, int_hexagon_S4_andi_asl_ri>;
def : T_IRI_pat <S4_ori_asl_ri,  int_hexagon_S4_ori_asl_ri>;
def : T_IRI_pat <S4_addi_asl_ri, int_hexagon_S4_addi_asl_ri>;
def : T_IRI_pat <S4_subi_asl_ri, int_hexagon_S4_subi_asl_ri>;
def : T_IRI_pat <S4_andi_lsr_ri, int_hexagon_S4_andi_lsr_ri>;
def : T_IRI_pat <S4_ori_lsr_ri,  int_hexagon_S4_ori_lsr_ri>;
def : T_IRI_pat <S4_addi_lsr_ri, int_hexagon_S4_addi_lsr_ri>;
def : T_IRI_pat <S4_subi_lsr_ri, int_hexagon_S4_subi_lsr_ri>;

// Split bitfield
def : T_RI_pat <A4_bitspliti, int_hexagon_A4_bitspliti>;
def : T_RR_pat <A4_bitsplit, int_hexagon_A4_bitsplit>;

def: T_RR_pat<S4_parity,   int_hexagon_S4_parity>;

def: T_RI_pat<S4_ntstbit_i,  int_hexagon_S4_ntstbit_i>;
def: T_RR_pat<S4_ntstbit_r,  int_hexagon_S4_ntstbit_r>;

def: T_RI_pat<S4_clbaddi,  int_hexagon_S4_clbaddi>;
def: T_PI_pat<S4_clbpaddi, int_hexagon_S4_clbpaddi>;
def: T_P_pat <S4_clbpnorm, int_hexagon_S4_clbpnorm>;

/********************************************************************
*            ALU32/ALU                                              *
*********************************************************************/

// ALU32 / ALU / Logical Operations.
def: T_RR_pat<A4_andn, int_hexagon_A4_andn>;
def: T_RR_pat<A4_orn,  int_hexagon_A4_orn>;

/********************************************************************
*            ALU32/PERM                                             *
*********************************************************************/

// Combine Words Into Doublewords.
def: T_RI_pat<A4_combineri, int_hexagon_A4_combineri, s8ExtPred>;
def: T_IR_pat<A4_combineir, int_hexagon_A4_combineir, s8ExtPred>;

/********************************************************************
*            ALU32/PRED                                             *
*********************************************************************/

def: T_RR_pat<A4_rcmpeq,  int_hexagon_A4_rcmpeq>;
def: T_RR_pat<A4_rcmpneq, int_hexagon_A4_rcmpneq>;

def: T_RI_pat<A4_rcmpeqi,  int_hexagon_A4_rcmpeqi>;
def: T_RI_pat<A4_rcmpneqi, int_hexagon_A4_rcmpneqi>;

/********************************************************************
*            XTYPE/ALU                                              *
*********************************************************************/

// Add And Accumulate.

def : T_RRI_pat <S4_addaddi, int_hexagon_S4_addaddi>;
def : T_RIR_pat <S4_subaddi, int_hexagon_S4_subaddi>;


// XTYPE / ALU / Logical-logical Words.
def : T_RRR_pat <M4_or_xor,   int_hexagon_M4_or_xor>;
def : T_RRR_pat <M4_and_xor,  int_hexagon_M4_and_xor>;
def : T_RRR_pat <M4_or_and,   int_hexagon_M4_or_and>;
def : T_RRR_pat <M4_and_and,  int_hexagon_M4_and_and>;
def : T_RRR_pat <M4_xor_and,  int_hexagon_M4_xor_and>;
def : T_RRR_pat <M4_or_or,    int_hexagon_M4_or_or>;
def : T_RRR_pat <M4_and_or,   int_hexagon_M4_and_or>;
def : T_RRR_pat <M4_xor_or,   int_hexagon_M4_xor_or>;
def : T_RRR_pat <M4_or_andn,  int_hexagon_M4_or_andn>;
def : T_RRR_pat <M4_and_andn, int_hexagon_M4_and_andn>;
def : T_RRR_pat <M4_xor_andn, int_hexagon_M4_xor_andn>;

def : T_RRI_pat <S4_or_andi, int_hexagon_S4_or_andi>;
def : T_RRI_pat <S4_or_andix,  int_hexagon_S4_or_andix>;
def : T_RRI_pat <S4_or_ori, int_hexagon_S4_or_ori>;

// Modulo wrap.
def : T_RR_pat <A4_modwrapu, int_hexagon_A4_modwrapu>;

// Arithmetic/Convergent round
// Rd=[cround|round](Rs,Rt)[:sat]
// Rd=[cround|round](Rs,#u5)[:sat]
def : T_RI_pat <A4_cround_ri, int_hexagon_A4_cround_ri>;
def : T_RR_pat <A4_cround_rr, int_hexagon_A4_cround_rr>;

def : T_RI_pat <A4_round_ri, int_hexagon_A4_round_ri>;
def : T_RR_pat <A4_round_rr, int_hexagon_A4_round_rr>;

def : T_RI_pat <A4_round_ri_sat, int_hexagon_A4_round_ri_sat>;
def : T_RR_pat <A4_round_rr_sat, int_hexagon_A4_round_rr_sat>;

def : T_P_pat <A2_roundsat, int_hexagon_A2_roundsat>;

class qi_neg_ALU32_sisi<string opc, Intrinsic IntID>
  : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = !", !strconcat(opc , "($src1, $src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class qi_neg_ALU32_sis10<string opc, Intrinsic IntID>
  : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$src1, s10Imm:$src2),
             !strconcat("$dst = !", !strconcat(opc , "($src1, #$src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class qi_neg_ALU32_siu9<string opc, Intrinsic IntID>
  : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$src1, u9Imm:$src2),
             !strconcat("$dst = !", !strconcat(opc , "($src1, #$src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

//
// SInst Classes.
//
class qi_neg_SInst_qiqi<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = !", !strconcat(opc , "($src1, $src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class qi_SInst_qi_andqiqi_neg<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2,
                                     IntRegs:$src3),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, and($src2, !$src3)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2,
                                         IntRegs:$src3))]>;

class qi_SInst_qi_andqiqi<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2,
                                     IntRegs:$src3),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, and($src2, $src3)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2,
                                         IntRegs:$src3))]>;

class qi_SInst_qi_orqiqi_neg<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2,
                                     IntRegs:$src3),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, or($src2, !$src3)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2,
                                         IntRegs:$src3))]>;

class qi_SInst_qi_orqiqi<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2,
                                     IntRegs:$src3),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, or($src2, $src3)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2,
                                         IntRegs:$src3))]>;

/********************************************************************
*            ALU32/PRED                                             *
*********************************************************************/

// ALU32 / PRED / Conditional Shift Halfword.
// ALU32 / PRED / Conditional Sign Extend.
// ALU32 / PRED / Conditional Zero Extend.
// ALU32 / PRED / Compare.
def Hexagon_C4_cmpltei : qi_neg_ALU32_sis10 <"cmp.gt", int_hexagon_C4_cmpltei>;
def Hexagon_C4_cmplte  : qi_neg_ALU32_sisi  <"cmp.gt", int_hexagon_C4_cmplte>;
def Hexagon_C4_cmplteu : qi_neg_ALU32_sisi  <"cmp.gtu",int_hexagon_C4_cmplteu>;

def: T_RI_pat<C4_cmpneqi, int_hexagon_C4_cmpneqi>;
def: T_RI_pat<C4_cmpltei, int_hexagon_C4_cmpltei>;
def: T_RI_pat<C4_cmplteui, int_hexagon_C4_cmplteui>;


/********************************************************************
*            CR                                                     *
*********************************************************************/

// CR / Corner Detection Acceleration.
def Hexagon_C4_fastcorner9:
  qi_SInst_qiqi<"fastcorner9", int_hexagon_C4_fastcorner9>;
def Hexagon_C4_fastcorner9_not:
  qi_neg_SInst_qiqi<"fastcorner9",int_hexagon_C4_fastcorner9_not>;

// CR / Logical Operations On Predicates.
def Hexagon_C4_and_andn:
  qi_SInst_qi_andqiqi_neg         <"and",      int_hexagon_C4_and_andn>;
def Hexagon_C4_and_and:
  qi_SInst_qi_andqiqi             <"and",      int_hexagon_C4_and_and>;
def Hexagon_C4_and_orn:
  qi_SInst_qi_orqiqi_neg          <"and",      int_hexagon_C4_and_orn>;
def Hexagon_C4_and_or:
  qi_SInst_qi_orqiqi              <"and",      int_hexagon_C4_and_or>;
def Hexagon_C4_or_andn:
  qi_SInst_qi_andqiqi_neg         <"or",       int_hexagon_C4_or_andn>;
def Hexagon_C4_or_and:
  qi_SInst_qi_andqiqi             <"or",       int_hexagon_C4_or_and>;
def Hexagon_C4_or_orn:
  qi_SInst_qi_orqiqi_neg          <"or",       int_hexagon_C4_or_orn>;
def Hexagon_C4_or_or:
  qi_SInst_qi_orqiqi              <"or",       int_hexagon_C4_or_or>;
