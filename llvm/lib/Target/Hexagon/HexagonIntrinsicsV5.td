//===- HexagonIntrinsicsV5.td - V4 Instruction intrinsics --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def : T_FF_pat<F2_sfadd, int_hexagon_F2_sfadd>;
def : T_FF_pat<F2_sfsub, int_hexagon_F2_sfsub>;
def : T_FF_pat<F2_sfmpy, int_hexagon_F2_sfmpy>;
def : T_FF_pat<F2_sfmax, int_hexagon_F2_sfmax>;
def : T_FF_pat<F2_sfmin, int_hexagon_F2_sfmin>;

def : T_FF_pat<F2_sffixupn, int_hexagon_F2_sffixupn>;
def : T_FF_pat<F2_sffixupd, int_hexagon_F2_sffixupd>;
def : T_F_pat <F2_sffixupr, int_hexagon_F2_sffixupr>;

def: qi_CRInst_qiqi_pat<C4_fastcorner9,     int_hexagon_C4_fastcorner9>;
def: qi_CRInst_qiqi_pat<C4_fastcorner9_not, int_hexagon_C4_fastcorner9_not>;

def : T_P_pat <S5_popcountp, int_hexagon_S5_popcountp>;
def : T_PI_pat <S5_asrhub_sat, int_hexagon_S5_asrhub_sat>;

def : T_PI_pat <S2_asr_i_p_rnd, int_hexagon_S2_asr_i_p_rnd>;
def : T_PI_pat <S2_asr_i_p_rnd_goodsyntax,
                int_hexagon_S2_asr_i_p_rnd_goodsyntax>;

def : T_PI_pat <S5_asrhub_rnd_sat_goodsyntax,
                int_hexagon_S5_asrhub_rnd_sat_goodsyntax>;

def : T_FFF_pat <F2_sffma, int_hexagon_F2_sffma>;
def : T_FFF_pat <F2_sffms, int_hexagon_F2_sffms>;
def : T_FFF_pat <F2_sffma_lib, int_hexagon_F2_sffma_lib>;
def : T_FFF_pat <F2_sffms_lib, int_hexagon_F2_sffms_lib>;
def : T_FFFQ_pat <F2_sffma_sc, int_hexagon_F2_sffma_sc>;

// Compare floating-point value
def : T_FF_pat <F2_sfcmpge, int_hexagon_F2_sfcmpge>;
def : T_FF_pat <F2_sfcmpuo, int_hexagon_F2_sfcmpuo>;
def : T_FF_pat <F2_sfcmpeq, int_hexagon_F2_sfcmpeq>;
def : T_FF_pat <F2_sfcmpgt, int_hexagon_F2_sfcmpgt>;

def : T_DD_pat <F2_dfcmpeq, int_hexagon_F2_dfcmpeq>;
def : T_DD_pat <F2_dfcmpgt, int_hexagon_F2_dfcmpgt>;
def : T_DD_pat <F2_dfcmpge, int_hexagon_F2_dfcmpge>;
def : T_DD_pat <F2_dfcmpuo, int_hexagon_F2_dfcmpuo>;

// Create floating-point value
def : T_I_pat <F2_sfimm_p, int_hexagon_F2_sfimm_p>;
def : T_I_pat <F2_sfimm_n, int_hexagon_F2_sfimm_n>;
def : T_I_pat <F2_dfimm_p, int_hexagon_F2_dfimm_p>;
def : T_I_pat <F2_dfimm_n, int_hexagon_F2_dfimm_n>;

def : T_DI_pat <F2_dfclass, int_hexagon_F2_dfclass>;
def : T_FI_pat <F2_sfclass, int_hexagon_F2_sfclass>;
def : T_F_pat <F2_conv_sf2df, int_hexagon_F2_conv_sf2df>;
def : T_D_pat <F2_conv_df2sf, int_hexagon_F2_conv_df2sf>;
def : T_R_pat <F2_conv_uw2sf, int_hexagon_F2_conv_uw2sf>;
def : T_R_pat <F2_conv_uw2df, int_hexagon_F2_conv_uw2df>;
def : T_R_pat <F2_conv_w2sf,  int_hexagon_F2_conv_w2sf>;
def : T_R_pat <F2_conv_w2df,  int_hexagon_F2_conv_w2df>;
def : T_P_pat <F2_conv_ud2sf, int_hexagon_F2_conv_ud2sf>;
def : T_P_pat <F2_conv_ud2df, int_hexagon_F2_conv_ud2df>;
def : T_P_pat <F2_conv_d2sf,  int_hexagon_F2_conv_d2sf>;
def : T_P_pat <F2_conv_d2df,  int_hexagon_F2_conv_d2df>;
def : T_F_pat <F2_conv_sf2uw, int_hexagon_F2_conv_sf2uw>;
def : T_F_pat <F2_conv_sf2w,  int_hexagon_F2_conv_sf2w>;
def : T_F_pat <F2_conv_sf2ud, int_hexagon_F2_conv_sf2ud>;
def : T_F_pat <F2_conv_sf2d,  int_hexagon_F2_conv_sf2d>;
def : T_D_pat <F2_conv_df2uw, int_hexagon_F2_conv_df2uw>;
def : T_D_pat <F2_conv_df2w,  int_hexagon_F2_conv_df2w>;
def : T_D_pat <F2_conv_df2ud, int_hexagon_F2_conv_df2ud>;
def : T_D_pat <F2_conv_df2d,  int_hexagon_F2_conv_df2d>;
def : T_F_pat <F2_conv_sf2uw_chop, int_hexagon_F2_conv_sf2uw_chop>;
def : T_F_pat <F2_conv_sf2w_chop,  int_hexagon_F2_conv_sf2w_chop>;
def : T_F_pat <F2_conv_sf2ud_chop, int_hexagon_F2_conv_sf2ud_chop>;
def : T_F_pat <F2_conv_sf2d_chop,  int_hexagon_F2_conv_sf2d_chop>;
def : T_D_pat <F2_conv_df2uw_chop, int_hexagon_F2_conv_df2uw_chop>;
def : T_D_pat <F2_conv_df2w_chop,  int_hexagon_F2_conv_df2w_chop>;
def : T_D_pat <F2_conv_df2ud_chop, int_hexagon_F2_conv_df2ud_chop>;
def : T_D_pat <F2_conv_df2d_chop,  int_hexagon_F2_conv_df2d_chop>;

class qi_ALU64_dfdf<string opc, Intrinsic IntID>
  : ALU64_rr<(outs PredRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
           !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
           [(set PredRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2))]>;

class qi_ALU64_dfu5<string opc, Intrinsic IntID>
  : ALU64_ri<(outs PredRegs:$dst), (ins DoubleRegs:$src1, u5Imm:$src2),
           !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
           [(set PredRegs:$dst, (IntID DoubleRegs:$src1, imm:$src2))]>;

class qi_SInst_sfsf<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class qi_SInst_sfu5<string opc, Intrinsic IntID>
  : MInst<(outs PredRegs:$dst), (ins IntRegs:$src1, u5Imm:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class di_MInst_diu4_rnd<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, u4Imm:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2):rnd")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, imm:$src2))]>;

def HEXAGON_M5_vrmpybuu:
    di_MInst_didi <"vrmpybu", int_hexagon_M5_vrmpybuu>;
def HEXAGON_M5_vrmacbuu:
    di_MInst_dididi_acc <"vrmpybu", int_hexagon_M5_vrmacbuu>;
def HEXAGON_M5_vrmpybsu:
    di_MInst_didi <"vrmpybsu", int_hexagon_M5_vrmpybsu>;
def HEXAGON_M5_vrmacbsu:
    di_MInst_dididi_acc <"vrmpybsu", int_hexagon_M5_vrmacbsu>;
def HEXAGON_M5_vmpybuu:
    di_MInst_sisi <"vmpybu", int_hexagon_M5_vmpybuu>;
def HEXAGON_M5_vmpybsu:
    di_MInst_sisi <"vmpybsu", int_hexagon_M5_vmpybsu>;
def HEXAGON_M5_vmacbuu:
    di_MInst_disisi_acc <"vmpybu", int_hexagon_M5_vmacbuu>;
def HEXAGON_M5_vmacbsu:
    di_MInst_disisi_acc <"vmpybsu", int_hexagon_M5_vmacbsu>;
def HEXAGON_M5_vdmpybsu:
    di_MInst_didi_sat <"vdmpybsu", int_hexagon_M5_vdmpybsu>;
def HEXAGON_M5_vdmacbsu:
    di_MInst_dididi_acc_sat <"vdmpybsu", int_hexagon_M5_vdmacbsu>;
def HEXAGON_A5_vaddhubs:
    si_SInst_didi_sat <"vaddhub", int_hexagon_A5_vaddhubs>;
def HEXAGON_S5_vasrhrnd_goodsyntax:
    di_MInst_diu4_rnd <"vasrh", int_hexagon_S5_vasrhrnd_goodsyntax>;
