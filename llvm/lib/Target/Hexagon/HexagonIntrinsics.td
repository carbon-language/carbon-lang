//===-- HexagonIntrinsics.td - Instruction intrinsics ------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This is populated based on the following specs:
// Hexagon V2 Architecture
// Application-Level Specification
// 80-V9418-8 Rev. B
// March 4, 2008
//===----------------------------------------------------------------------===//

class T_I_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID imm:$Is),
         (MI imm:$Is)>;

class T_R_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rs),
         (MI I32:$Rs)>;

class T_P_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs),
         (MI DoubleRegs:$Rs)>;

class T_II_pat <InstHexagon MI, Intrinsic IntID, PatFrag Imm1, PatFrag Imm2>
  : Pat<(IntID Imm1:$Is, Imm2:$It),
        (MI Imm1:$Is, Imm2:$It)>;

class T_RI_pat <InstHexagon MI, Intrinsic IntID, PatLeaf ImmPred = PatLeaf<(i32 imm)>>
  : Pat<(IntID I32:$Rs, ImmPred:$It),
        (MI I32:$Rs, ImmPred:$It)>;

class T_IR_pat <InstHexagon MI, Intrinsic IntID, PatFrag ImmPred = PatLeaf<(i32 imm)>>
  : Pat<(IntID ImmPred:$Is, I32:$Rt),
        (MI ImmPred:$Is, I32:$Rt)>;

class T_PI_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID I64:$Rs, imm:$It),
        (MI DoubleRegs:$Rs, imm:$It)>;

class T_RP_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID I32:$Rs, I64:$Rt),
        (MI I32:$Rs, DoubleRegs:$Rt)>;

class T_RR_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rs, I32:$Rt),
         (MI I32:$Rs, I32:$Rt)>;

class T_PP_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs, I64:$Rt),
         (MI DoubleRegs:$Rs, DoubleRegs:$Rt)>;

class T_QII_pat <InstHexagon MI, Intrinsic IntID, PatFrag Imm1, PatFrag Imm2>
  : Pat <(IntID (i32 PredRegs:$Ps), Imm1:$Is, Imm2:$It),
         (MI PredRegs:$Ps, Imm1:$Is, Imm2:$It)>;

class T_QRI_pat <InstHexagon MI, Intrinsic IntID, PatFrag ImmPred>
  : Pat <(IntID (i32 PredRegs:$Ps), I32:$Rs, ImmPred:$Is),
         (MI PredRegs:$Ps, I32:$Rs, ImmPred:$Is)>;

class T_QIR_pat <InstHexagon MI, Intrinsic IntID, PatFrag ImmPred>
  : Pat <(IntID (i32 PredRegs:$Ps), ImmPred:$Is, I32:$Rs),
         (MI PredRegs:$Ps, ImmPred:$Is, I32:$Rs)>;

class T_RRI_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rs, I32:$Rt, imm:$Iu),
         (MI I32:$Rs, I32:$Rt, imm:$Iu)>;

class T_RII_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rs, imm:$It, imm:$Iu),
         (MI I32:$Rs, imm:$It, imm:$Iu)>;

class T_IRI_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID imm:$It, I32:$Rs, imm:$Iu),
         (MI imm:$It, I32:$Rs, imm:$Iu)>;

class T_IRR_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID imm:$Is, I32:$Rs, I32:$Rt),
         (MI imm:$Is, I32:$Rs, I32:$Rt)>;

class T_RIR_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rs, imm:$Is, I32:$Rt),
         (MI I32:$Rs, imm:$Is, I32:$Rt)>;

class T_RRR_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rs, I32:$Rt, I32:$Ru),
         (MI I32:$Rs, I32:$Rt, I32:$Ru)>;

class T_PPI_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs, I64:$Rt, imm:$Iu),
         (MI DoubleRegs:$Rs, DoubleRegs:$Rt, imm:$Iu)>;

class T_PII_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs, imm:$It, imm:$Iu),
         (MI DoubleRegs:$Rs, imm:$It, imm:$Iu)>;

class T_PPP_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs, I64:$Rt, I64:$Ru),
         (MI DoubleRegs:$Rs, DoubleRegs:$Rt, DoubleRegs:$Ru)>;

class T_PPR_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs, I64:$Rt, I32:$Ru),
         (MI DoubleRegs:$Rs, DoubleRegs:$Rt, I32:$Ru)>;

class T_PRR_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs, I32:$Rt, I32:$Ru),
         (MI DoubleRegs:$Rs, I32:$Rt, I32:$Ru)>;

class T_PR_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs, I32:$Rt),
         (MI DoubleRegs:$Rs, I32:$Rt)>;

class T_D_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID (F64:$Rs)),
        (MI (F64:$Rs))>;

class T_DI_pat <InstHexagon MI, Intrinsic IntID,
                PatLeaf ImmPred = PatLeaf<(i32 imm)>>
  : Pat<(IntID F64:$Rs, ImmPred:$It),
        (MI F64:$Rs, ImmPred:$It)>;

class T_F_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID F32:$Rs),
        (MI F32:$Rs)>;

class T_FI_pat <InstHexagon MI, Intrinsic IntID,
                 PatLeaf ImmPred = PatLeaf<(i32 imm)>>
  : Pat<(IntID F32:$Rs, ImmPred:$It),
        (MI F32:$Rs, ImmPred:$It)>;

class T_FF_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID F32:$Rs, F32:$Rt),
        (MI F32:$Rs, F32:$Rt)>;

class T_DD_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID F64:$Rs, F64:$Rt),
        (MI F64:$Rs, F64:$Rt)>;

class T_FFF_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID F32:$Rs, F32:$Rt, F32:$Ru),
        (MI F32:$Rs, F32:$Rt, F32:$Ru)>;

class T_FFFQ_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID F32:$Rs, F32:$Rt, F32:$Ru, (i32 PredRegs:$Rx)),
         (MI F32:$Rs, F32:$Rt, F32:$Ru, PredRegs:$Rx)>;

//===----------------------------------------------------------------------===//
// MPYS / Multipy signed/unsigned halfwords
//Rd=mpy[u](Rs.[H|L],Rt.[H|L])[:<<1][:rnd][:sat]
//===----------------------------------------------------------------------===//

def : T_RR_pat <M2_mpy_ll_s1, int_hexagon_M2_mpy_ll_s1>;
def : T_RR_pat <M2_mpy_ll_s0, int_hexagon_M2_mpy_ll_s0>;
def : T_RR_pat <M2_mpy_lh_s1, int_hexagon_M2_mpy_lh_s1>;
def : T_RR_pat <M2_mpy_lh_s0, int_hexagon_M2_mpy_lh_s0>;
def : T_RR_pat <M2_mpy_hl_s1, int_hexagon_M2_mpy_hl_s1>;
def : T_RR_pat <M2_mpy_hl_s0, int_hexagon_M2_mpy_hl_s0>;
def : T_RR_pat <M2_mpy_hh_s1, int_hexagon_M2_mpy_hh_s1>;
def : T_RR_pat <M2_mpy_hh_s0, int_hexagon_M2_mpy_hh_s0>;

def : T_RR_pat <M2_mpyu_ll_s1, int_hexagon_M2_mpyu_ll_s1>;
def : T_RR_pat <M2_mpyu_ll_s0, int_hexagon_M2_mpyu_ll_s0>;
def : T_RR_pat <M2_mpyu_lh_s1, int_hexagon_M2_mpyu_lh_s1>;
def : T_RR_pat <M2_mpyu_lh_s0, int_hexagon_M2_mpyu_lh_s0>;
def : T_RR_pat <M2_mpyu_hl_s1, int_hexagon_M2_mpyu_hl_s1>;
def : T_RR_pat <M2_mpyu_hl_s0, int_hexagon_M2_mpyu_hl_s0>;
def : T_RR_pat <M2_mpyu_hh_s1, int_hexagon_M2_mpyu_hh_s1>;
def : T_RR_pat <M2_mpyu_hh_s0, int_hexagon_M2_mpyu_hh_s0>;

def : T_RR_pat <M2_mpy_sat_ll_s1, int_hexagon_M2_mpy_sat_ll_s1>;
def : T_RR_pat <M2_mpy_sat_ll_s0, int_hexagon_M2_mpy_sat_ll_s0>;
def : T_RR_pat <M2_mpy_sat_lh_s1, int_hexagon_M2_mpy_sat_lh_s1>;
def : T_RR_pat <M2_mpy_sat_lh_s0, int_hexagon_M2_mpy_sat_lh_s0>;
def : T_RR_pat <M2_mpy_sat_hl_s1, int_hexagon_M2_mpy_sat_hl_s1>;
def : T_RR_pat <M2_mpy_sat_hl_s0, int_hexagon_M2_mpy_sat_hl_s0>;
def : T_RR_pat <M2_mpy_sat_hh_s1, int_hexagon_M2_mpy_sat_hh_s1>;
def : T_RR_pat <M2_mpy_sat_hh_s0, int_hexagon_M2_mpy_sat_hh_s0>;

def : T_RR_pat <M2_mpy_rnd_ll_s1, int_hexagon_M2_mpy_rnd_ll_s1>;
def : T_RR_pat <M2_mpy_rnd_ll_s0, int_hexagon_M2_mpy_rnd_ll_s0>;
def : T_RR_pat <M2_mpy_rnd_lh_s1, int_hexagon_M2_mpy_rnd_lh_s1>;
def : T_RR_pat <M2_mpy_rnd_lh_s0, int_hexagon_M2_mpy_rnd_lh_s0>;
def : T_RR_pat <M2_mpy_rnd_hl_s1, int_hexagon_M2_mpy_rnd_hl_s1>;
def : T_RR_pat <M2_mpy_rnd_hl_s0, int_hexagon_M2_mpy_rnd_hl_s0>;
def : T_RR_pat <M2_mpy_rnd_hh_s1, int_hexagon_M2_mpy_rnd_hh_s1>;
def : T_RR_pat <M2_mpy_rnd_hh_s0, int_hexagon_M2_mpy_rnd_hh_s0>;

def : T_RR_pat <M2_mpy_sat_rnd_ll_s1, int_hexagon_M2_mpy_sat_rnd_ll_s1>;
def : T_RR_pat <M2_mpy_sat_rnd_ll_s0, int_hexagon_M2_mpy_sat_rnd_ll_s0>;
def : T_RR_pat <M2_mpy_sat_rnd_lh_s1, int_hexagon_M2_mpy_sat_rnd_lh_s1>;
def : T_RR_pat <M2_mpy_sat_rnd_lh_s0, int_hexagon_M2_mpy_sat_rnd_lh_s0>;
def : T_RR_pat <M2_mpy_sat_rnd_hl_s1, int_hexagon_M2_mpy_sat_rnd_hl_s1>;
def : T_RR_pat <M2_mpy_sat_rnd_hl_s0, int_hexagon_M2_mpy_sat_rnd_hl_s0>;
def : T_RR_pat <M2_mpy_sat_rnd_hh_s1, int_hexagon_M2_mpy_sat_rnd_hh_s1>;
def : T_RR_pat <M2_mpy_sat_rnd_hh_s0, int_hexagon_M2_mpy_sat_rnd_hh_s0>;


//===----------------------------------------------------------------------===//
// MPYS / Multipy signed/unsigned halfwords and add/subtract the
// result from the accumulator.
//Rx [-+]= mpy[u](Rs.[H|L],Rt.[H|L])[:<<1][:sat]
//===----------------------------------------------------------------------===//

def : T_RRR_pat <M2_mpy_acc_ll_s1, int_hexagon_M2_mpy_acc_ll_s1>;
def : T_RRR_pat <M2_mpy_acc_ll_s0, int_hexagon_M2_mpy_acc_ll_s0>;
def : T_RRR_pat <M2_mpy_acc_lh_s1, int_hexagon_M2_mpy_acc_lh_s1>;
def : T_RRR_pat <M2_mpy_acc_lh_s0, int_hexagon_M2_mpy_acc_lh_s0>;
def : T_RRR_pat <M2_mpy_acc_hl_s1, int_hexagon_M2_mpy_acc_hl_s1>;
def : T_RRR_pat <M2_mpy_acc_hl_s0, int_hexagon_M2_mpy_acc_hl_s0>;
def : T_RRR_pat <M2_mpy_acc_hh_s1, int_hexagon_M2_mpy_acc_hh_s1>;
def : T_RRR_pat <M2_mpy_acc_hh_s0, int_hexagon_M2_mpy_acc_hh_s0>;

def : T_RRR_pat <M2_mpyu_acc_ll_s1, int_hexagon_M2_mpyu_acc_ll_s1>;
def : T_RRR_pat <M2_mpyu_acc_ll_s0, int_hexagon_M2_mpyu_acc_ll_s0>;
def : T_RRR_pat <M2_mpyu_acc_lh_s1, int_hexagon_M2_mpyu_acc_lh_s1>;
def : T_RRR_pat <M2_mpyu_acc_lh_s0, int_hexagon_M2_mpyu_acc_lh_s0>;
def : T_RRR_pat <M2_mpyu_acc_hl_s1, int_hexagon_M2_mpyu_acc_hl_s1>;
def : T_RRR_pat <M2_mpyu_acc_hl_s0, int_hexagon_M2_mpyu_acc_hl_s0>;
def : T_RRR_pat <M2_mpyu_acc_hh_s1, int_hexagon_M2_mpyu_acc_hh_s1>;
def : T_RRR_pat <M2_mpyu_acc_hh_s0, int_hexagon_M2_mpyu_acc_hh_s0>;

def : T_RRR_pat <M2_mpy_nac_ll_s1, int_hexagon_M2_mpy_nac_ll_s1>;
def : T_RRR_pat <M2_mpy_nac_ll_s0, int_hexagon_M2_mpy_nac_ll_s0>;
def : T_RRR_pat <M2_mpy_nac_lh_s1, int_hexagon_M2_mpy_nac_lh_s1>;
def : T_RRR_pat <M2_mpy_nac_lh_s0, int_hexagon_M2_mpy_nac_lh_s0>;
def : T_RRR_pat <M2_mpy_nac_hl_s1, int_hexagon_M2_mpy_nac_hl_s1>;
def : T_RRR_pat <M2_mpy_nac_hl_s0, int_hexagon_M2_mpy_nac_hl_s0>;
def : T_RRR_pat <M2_mpy_nac_hh_s1, int_hexagon_M2_mpy_nac_hh_s1>;
def : T_RRR_pat <M2_mpy_nac_hh_s0, int_hexagon_M2_mpy_nac_hh_s0>;

def : T_RRR_pat <M2_mpyu_nac_ll_s1, int_hexagon_M2_mpyu_nac_ll_s1>;
def : T_RRR_pat <M2_mpyu_nac_ll_s0, int_hexagon_M2_mpyu_nac_ll_s0>;
def : T_RRR_pat <M2_mpyu_nac_lh_s1, int_hexagon_M2_mpyu_nac_lh_s1>;
def : T_RRR_pat <M2_mpyu_nac_lh_s0, int_hexagon_M2_mpyu_nac_lh_s0>;
def : T_RRR_pat <M2_mpyu_nac_hl_s1, int_hexagon_M2_mpyu_nac_hl_s1>;
def : T_RRR_pat <M2_mpyu_nac_hl_s0, int_hexagon_M2_mpyu_nac_hl_s0>;
def : T_RRR_pat <M2_mpyu_nac_hh_s1, int_hexagon_M2_mpyu_nac_hh_s1>;
def : T_RRR_pat <M2_mpyu_nac_hh_s0, int_hexagon_M2_mpyu_nac_hh_s0>;

def : T_RRR_pat <M2_mpy_acc_sat_ll_s1, int_hexagon_M2_mpy_acc_sat_ll_s1>;
def : T_RRR_pat <M2_mpy_acc_sat_ll_s0, int_hexagon_M2_mpy_acc_sat_ll_s0>;
def : T_RRR_pat <M2_mpy_acc_sat_lh_s1, int_hexagon_M2_mpy_acc_sat_lh_s1>;
def : T_RRR_pat <M2_mpy_acc_sat_lh_s0, int_hexagon_M2_mpy_acc_sat_lh_s0>;
def : T_RRR_pat <M2_mpy_acc_sat_hl_s1, int_hexagon_M2_mpy_acc_sat_hl_s1>;
def : T_RRR_pat <M2_mpy_acc_sat_hl_s0, int_hexagon_M2_mpy_acc_sat_hl_s0>;
def : T_RRR_pat <M2_mpy_acc_sat_hh_s1, int_hexagon_M2_mpy_acc_sat_hh_s1>;
def : T_RRR_pat <M2_mpy_acc_sat_hh_s0, int_hexagon_M2_mpy_acc_sat_hh_s0>;

def : T_RRR_pat <M2_mpy_nac_sat_ll_s1, int_hexagon_M2_mpy_nac_sat_ll_s1>;
def : T_RRR_pat <M2_mpy_nac_sat_ll_s0, int_hexagon_M2_mpy_nac_sat_ll_s0>;
def : T_RRR_pat <M2_mpy_nac_sat_lh_s1, int_hexagon_M2_mpy_nac_sat_lh_s1>;
def : T_RRR_pat <M2_mpy_nac_sat_lh_s0, int_hexagon_M2_mpy_nac_sat_lh_s0>;
def : T_RRR_pat <M2_mpy_nac_sat_hl_s1, int_hexagon_M2_mpy_nac_sat_hl_s1>;
def : T_RRR_pat <M2_mpy_nac_sat_hl_s0, int_hexagon_M2_mpy_nac_sat_hl_s0>;
def : T_RRR_pat <M2_mpy_nac_sat_hh_s1, int_hexagon_M2_mpy_nac_sat_hh_s1>;
def : T_RRR_pat <M2_mpy_nac_sat_hh_s0, int_hexagon_M2_mpy_nac_sat_hh_s0>;


//===----------------------------------------------------------------------===//
// Multiply signed/unsigned halfwords with and without saturation and rounding
// into a 64-bits destination register.
//===----------------------------------------------------------------------===//

def : T_RR_pat <M2_mpyd_hh_s0, int_hexagon_M2_mpyd_hh_s0>;
def : T_RR_pat <M2_mpyd_hl_s0, int_hexagon_M2_mpyd_hl_s0>;
def : T_RR_pat <M2_mpyd_lh_s0, int_hexagon_M2_mpyd_lh_s0>;
def : T_RR_pat <M2_mpyd_ll_s0, int_hexagon_M2_mpyd_ll_s0>;
def : T_RR_pat <M2_mpyd_hh_s1, int_hexagon_M2_mpyd_hh_s1>;
def : T_RR_pat <M2_mpyd_hl_s1, int_hexagon_M2_mpyd_hl_s1>;
def : T_RR_pat <M2_mpyd_lh_s1, int_hexagon_M2_mpyd_lh_s1>;
def : T_RR_pat <M2_mpyd_ll_s1, int_hexagon_M2_mpyd_ll_s1>;

def : T_RR_pat <M2_mpyd_rnd_hh_s0, int_hexagon_M2_mpyd_rnd_hh_s0>;
def : T_RR_pat <M2_mpyd_rnd_hl_s0, int_hexagon_M2_mpyd_rnd_hl_s0>;
def : T_RR_pat <M2_mpyd_rnd_lh_s0, int_hexagon_M2_mpyd_rnd_lh_s0>;
def : T_RR_pat <M2_mpyd_rnd_ll_s0, int_hexagon_M2_mpyd_rnd_ll_s0>;
def : T_RR_pat <M2_mpyd_rnd_hh_s1, int_hexagon_M2_mpyd_rnd_hh_s1>;
def : T_RR_pat <M2_mpyd_rnd_hl_s1, int_hexagon_M2_mpyd_rnd_hl_s1>;
def : T_RR_pat <M2_mpyd_rnd_lh_s1, int_hexagon_M2_mpyd_rnd_lh_s1>;
def : T_RR_pat <M2_mpyd_rnd_ll_s1, int_hexagon_M2_mpyd_rnd_ll_s1>;

def : T_RR_pat <M2_mpyud_hh_s0, int_hexagon_M2_mpyud_hh_s0>;
def : T_RR_pat <M2_mpyud_hl_s0, int_hexagon_M2_mpyud_hl_s0>;
def : T_RR_pat <M2_mpyud_lh_s0, int_hexagon_M2_mpyud_lh_s0>;
def : T_RR_pat <M2_mpyud_ll_s0, int_hexagon_M2_mpyud_ll_s0>;
def : T_RR_pat <M2_mpyud_hh_s1, int_hexagon_M2_mpyud_hh_s1>;
def : T_RR_pat <M2_mpyud_hl_s1, int_hexagon_M2_mpyud_hl_s1>;
def : T_RR_pat <M2_mpyud_lh_s1, int_hexagon_M2_mpyud_lh_s1>;
def : T_RR_pat <M2_mpyud_ll_s1, int_hexagon_M2_mpyud_ll_s1>;

//===----------------------------------------------------------------------===//
// MPYS / Multipy signed/unsigned halfwords and add/subtract the
// result from the 64-bit destination register.
//Rxx [-+]= mpy[u](Rs.[H|L],Rt.[H|L])[:<<1][:sat]
//===----------------------------------------------------------------------===//

def : T_PRR_pat <M2_mpyd_acc_hh_s0, int_hexagon_M2_mpyd_acc_hh_s0>;
def : T_PRR_pat <M2_mpyd_acc_hl_s0, int_hexagon_M2_mpyd_acc_hl_s0>;
def : T_PRR_pat <M2_mpyd_acc_lh_s0, int_hexagon_M2_mpyd_acc_lh_s0>;
def : T_PRR_pat <M2_mpyd_acc_ll_s0, int_hexagon_M2_mpyd_acc_ll_s0>;

def : T_PRR_pat <M2_mpyd_acc_hh_s1, int_hexagon_M2_mpyd_acc_hh_s1>;
def : T_PRR_pat <M2_mpyd_acc_hl_s1, int_hexagon_M2_mpyd_acc_hl_s1>;
def : T_PRR_pat <M2_mpyd_acc_lh_s1, int_hexagon_M2_mpyd_acc_lh_s1>;
def : T_PRR_pat <M2_mpyd_acc_ll_s1, int_hexagon_M2_mpyd_acc_ll_s1>;

def : T_PRR_pat <M2_mpyd_nac_hh_s0, int_hexagon_M2_mpyd_nac_hh_s0>;
def : T_PRR_pat <M2_mpyd_nac_hl_s0, int_hexagon_M2_mpyd_nac_hl_s0>;
def : T_PRR_pat <M2_mpyd_nac_lh_s0, int_hexagon_M2_mpyd_nac_lh_s0>;
def : T_PRR_pat <M2_mpyd_nac_ll_s0, int_hexagon_M2_mpyd_nac_ll_s0>;

def : T_PRR_pat <M2_mpyd_nac_hh_s1, int_hexagon_M2_mpyd_nac_hh_s1>;
def : T_PRR_pat <M2_mpyd_nac_hl_s1, int_hexagon_M2_mpyd_nac_hl_s1>;
def : T_PRR_pat <M2_mpyd_nac_lh_s1, int_hexagon_M2_mpyd_nac_lh_s1>;
def : T_PRR_pat <M2_mpyd_nac_ll_s1, int_hexagon_M2_mpyd_nac_ll_s1>;

def : T_PRR_pat <M2_mpyud_acc_hh_s0, int_hexagon_M2_mpyud_acc_hh_s0>;
def : T_PRR_pat <M2_mpyud_acc_hl_s0, int_hexagon_M2_mpyud_acc_hl_s0>;
def : T_PRR_pat <M2_mpyud_acc_lh_s0, int_hexagon_M2_mpyud_acc_lh_s0>;
def : T_PRR_pat <M2_mpyud_acc_ll_s0, int_hexagon_M2_mpyud_acc_ll_s0>;

def : T_PRR_pat <M2_mpyud_acc_hh_s1, int_hexagon_M2_mpyud_acc_hh_s1>;
def : T_PRR_pat <M2_mpyud_acc_hl_s1, int_hexagon_M2_mpyud_acc_hl_s1>;
def : T_PRR_pat <M2_mpyud_acc_lh_s1, int_hexagon_M2_mpyud_acc_lh_s1>;
def : T_PRR_pat <M2_mpyud_acc_ll_s1, int_hexagon_M2_mpyud_acc_ll_s1>;

def : T_PRR_pat <M2_mpyud_nac_hh_s0, int_hexagon_M2_mpyud_nac_hh_s0>;
def : T_PRR_pat <M2_mpyud_nac_hl_s0, int_hexagon_M2_mpyud_nac_hl_s0>;
def : T_PRR_pat <M2_mpyud_nac_lh_s0, int_hexagon_M2_mpyud_nac_lh_s0>;
def : T_PRR_pat <M2_mpyud_nac_ll_s0, int_hexagon_M2_mpyud_nac_ll_s0>;

def : T_PRR_pat <M2_mpyud_nac_hh_s1, int_hexagon_M2_mpyud_nac_hh_s1>;
def : T_PRR_pat <M2_mpyud_nac_hl_s1, int_hexagon_M2_mpyud_nac_hl_s1>;
def : T_PRR_pat <M2_mpyud_nac_lh_s1, int_hexagon_M2_mpyud_nac_lh_s1>;
def : T_PRR_pat <M2_mpyud_nac_ll_s1, int_hexagon_M2_mpyud_nac_ll_s1>;


//===----------------------------------------------------------------------===//
// Add/Subtract halfword
// Rd=add(Rt.L,Rs.[HL])[:sat]
// Rd=sub(Rt.L,Rs.[HL])[:sat]
// Rd=add(Rt.[LH],Rs.[HL])[:sat][:<16]
// Rd=sub(Rt.[LH],Rs.[HL])[:sat][:<16]
//===----------------------------------------------------------------------===//

//Rd=add(Rt.L,Rs.[LH])
def : T_RR_pat <A2_addh_l16_ll,     int_hexagon_A2_addh_l16_ll>;
def : T_RR_pat <A2_addh_l16_hl,     int_hexagon_A2_addh_l16_hl>;

//Rd=add(Rt.L,Rs.[LH]):sat
def : T_RR_pat <A2_addh_l16_sat_ll, int_hexagon_A2_addh_l16_sat_ll>;
def : T_RR_pat <A2_addh_l16_sat_hl, int_hexagon_A2_addh_l16_sat_hl>;

//Rd=sub(Rt.L,Rs.[LH])
def : T_RR_pat <A2_subh_l16_ll,     int_hexagon_A2_subh_l16_ll>;
def : T_RR_pat <A2_subh_l16_hl,     int_hexagon_A2_subh_l16_hl>;

//Rd=sub(Rt.L,Rs.[LH]):sat
def : T_RR_pat <A2_subh_l16_sat_ll, int_hexagon_A2_subh_l16_sat_ll>;
def : T_RR_pat <A2_subh_l16_sat_hl, int_hexagon_A2_subh_l16_sat_hl>;

//Rd=add(Rt.[LH],Rs.[LH]):<<16
def : T_RR_pat <A2_addh_h16_ll,     int_hexagon_A2_addh_h16_ll>;
def : T_RR_pat <A2_addh_h16_lh,     int_hexagon_A2_addh_h16_lh>;
def : T_RR_pat <A2_addh_h16_hl,     int_hexagon_A2_addh_h16_hl>;
def : T_RR_pat <A2_addh_h16_hh,     int_hexagon_A2_addh_h16_hh>;

//Rd=sub(Rt.[LH],Rs.[LH]):<<16
def : T_RR_pat <A2_subh_h16_ll,     int_hexagon_A2_subh_h16_ll>;
def : T_RR_pat <A2_subh_h16_lh,     int_hexagon_A2_subh_h16_lh>;
def : T_RR_pat <A2_subh_h16_hl,     int_hexagon_A2_subh_h16_hl>;
def : T_RR_pat <A2_subh_h16_hh,     int_hexagon_A2_subh_h16_hh>;

//Rd=add(Rt.[LH],Rs.[LH]):sat:<<16
def : T_RR_pat <A2_addh_h16_sat_ll, int_hexagon_A2_addh_h16_sat_ll>;
def : T_RR_pat <A2_addh_h16_sat_lh, int_hexagon_A2_addh_h16_sat_lh>;
def : T_RR_pat <A2_addh_h16_sat_hl, int_hexagon_A2_addh_h16_sat_hl>;
def : T_RR_pat <A2_addh_h16_sat_hh, int_hexagon_A2_addh_h16_sat_hh>;

//Rd=sub(Rt.[LH],Rs.[LH]):sat:<<16
def : T_RR_pat <A2_subh_h16_sat_ll, int_hexagon_A2_subh_h16_sat_ll>;
def : T_RR_pat <A2_subh_h16_sat_lh, int_hexagon_A2_subh_h16_sat_lh>;
def : T_RR_pat <A2_subh_h16_sat_hl, int_hexagon_A2_subh_h16_sat_hl>;
def : T_RR_pat <A2_subh_h16_sat_hh, int_hexagon_A2_subh_h16_sat_hh>;

// ALU64 / ALU / min max
def : T_RR_pat<A2_max,  int_hexagon_A2_max>;
def : T_RR_pat<A2_min,  int_hexagon_A2_min>;
def : T_RR_pat<A2_maxu, int_hexagon_A2_maxu>;
def : T_RR_pat<A2_minu, int_hexagon_A2_minu>;

// Shift and accumulate
def : T_RRI_pat <S2_asr_i_r_nac,  int_hexagon_S2_asr_i_r_nac>;
def : T_RRI_pat <S2_lsr_i_r_nac,  int_hexagon_S2_lsr_i_r_nac>;
def : T_RRI_pat <S2_asl_i_r_nac,  int_hexagon_S2_asl_i_r_nac>;
def : T_RRI_pat <S2_asr_i_r_acc,  int_hexagon_S2_asr_i_r_acc>;
def : T_RRI_pat <S2_lsr_i_r_acc,  int_hexagon_S2_lsr_i_r_acc>;
def : T_RRI_pat <S2_asl_i_r_acc,  int_hexagon_S2_asl_i_r_acc>;

def : T_RRI_pat <S2_asr_i_r_and,  int_hexagon_S2_asr_i_r_and>;
def : T_RRI_pat <S2_lsr_i_r_and,  int_hexagon_S2_lsr_i_r_and>;
def : T_RRI_pat <S2_asl_i_r_and,  int_hexagon_S2_asl_i_r_and>;
def : T_RRI_pat <S2_asr_i_r_or,   int_hexagon_S2_asr_i_r_or>;
def : T_RRI_pat <S2_lsr_i_r_or,   int_hexagon_S2_lsr_i_r_or>;
def : T_RRI_pat <S2_asl_i_r_or,   int_hexagon_S2_asl_i_r_or>;
def : T_RRI_pat <S2_lsr_i_r_xacc, int_hexagon_S2_lsr_i_r_xacc>;
def : T_RRI_pat <S2_asl_i_r_xacc, int_hexagon_S2_asl_i_r_xacc>;

def : T_PPI_pat <S2_asr_i_p_nac,  int_hexagon_S2_asr_i_p_nac>;
def : T_PPI_pat <S2_lsr_i_p_nac,  int_hexagon_S2_lsr_i_p_nac>;
def : T_PPI_pat <S2_asl_i_p_nac,  int_hexagon_S2_asl_i_p_nac>;
def : T_PPI_pat <S2_asr_i_p_acc,  int_hexagon_S2_asr_i_p_acc>;
def : T_PPI_pat <S2_lsr_i_p_acc,  int_hexagon_S2_lsr_i_p_acc>;
def : T_PPI_pat <S2_asl_i_p_acc,  int_hexagon_S2_asl_i_p_acc>;

def : T_PPI_pat <S2_asr_i_p_and,  int_hexagon_S2_asr_i_p_and>;
def : T_PPI_pat <S2_lsr_i_p_and,  int_hexagon_S2_lsr_i_p_and>;
def : T_PPI_pat <S2_asl_i_p_and,  int_hexagon_S2_asl_i_p_and>;
def : T_PPI_pat <S2_asr_i_p_or,   int_hexagon_S2_asr_i_p_or>;
def : T_PPI_pat <S2_lsr_i_p_or,   int_hexagon_S2_lsr_i_p_or>;
def : T_PPI_pat <S2_asl_i_p_or,   int_hexagon_S2_asl_i_p_or>;
def : T_PPI_pat <S2_lsr_i_p_xacc, int_hexagon_S2_lsr_i_p_xacc>;
def : T_PPI_pat <S2_asl_i_p_xacc, int_hexagon_S2_asl_i_p_xacc>;

def : T_RRR_pat <S2_asr_r_r_nac,  int_hexagon_S2_asr_r_r_nac>;
def : T_RRR_pat <S2_lsr_r_r_nac,  int_hexagon_S2_lsr_r_r_nac>;
def : T_RRR_pat <S2_asl_r_r_nac,  int_hexagon_S2_asl_r_r_nac>;
def : T_RRR_pat <S2_lsl_r_r_nac,  int_hexagon_S2_lsl_r_r_nac>;
def : T_RRR_pat <S2_asr_r_r_acc,  int_hexagon_S2_asr_r_r_acc>;
def : T_RRR_pat <S2_lsr_r_r_acc,  int_hexagon_S2_lsr_r_r_acc>;
def : T_RRR_pat <S2_asl_r_r_acc,  int_hexagon_S2_asl_r_r_acc>;
def : T_RRR_pat <S2_lsl_r_r_acc,  int_hexagon_S2_lsl_r_r_acc>;

def : T_RRR_pat <S2_asr_r_r_and,  int_hexagon_S2_asr_r_r_and>;
def : T_RRR_pat <S2_lsr_r_r_and,  int_hexagon_S2_lsr_r_r_and>;
def : T_RRR_pat <S2_asl_r_r_and,  int_hexagon_S2_asl_r_r_and>;
def : T_RRR_pat <S2_lsl_r_r_and,  int_hexagon_S2_lsl_r_r_and>;
def : T_RRR_pat <S2_asr_r_r_or,   int_hexagon_S2_asr_r_r_or>;
def : T_RRR_pat <S2_lsr_r_r_or,   int_hexagon_S2_lsr_r_r_or>;
def : T_RRR_pat <S2_asl_r_r_or,   int_hexagon_S2_asl_r_r_or>;
def : T_RRR_pat <S2_lsl_r_r_or,   int_hexagon_S2_lsl_r_r_or>;

def : T_PPR_pat <S2_asr_r_p_nac,  int_hexagon_S2_asr_r_p_nac>;
def : T_PPR_pat <S2_lsr_r_p_nac,  int_hexagon_S2_lsr_r_p_nac>;
def : T_PPR_pat <S2_asl_r_p_nac,  int_hexagon_S2_asl_r_p_nac>;
def : T_PPR_pat <S2_lsl_r_p_nac,  int_hexagon_S2_lsl_r_p_nac>;
def : T_PPR_pat <S2_asr_r_p_acc,  int_hexagon_S2_asr_r_p_acc>;
def : T_PPR_pat <S2_lsr_r_p_acc,  int_hexagon_S2_lsr_r_p_acc>;
def : T_PPR_pat <S2_asl_r_p_acc,  int_hexagon_S2_asl_r_p_acc>;
def : T_PPR_pat <S2_lsl_r_p_acc,  int_hexagon_S2_lsl_r_p_acc>;

def : T_PPR_pat <S2_asr_r_p_and,  int_hexagon_S2_asr_r_p_and>;
def : T_PPR_pat <S2_lsr_r_p_and,  int_hexagon_S2_lsr_r_p_and>;
def : T_PPR_pat <S2_asl_r_p_and,  int_hexagon_S2_asl_r_p_and>;
def : T_PPR_pat <S2_lsl_r_p_and,  int_hexagon_S2_lsl_r_p_and>;
def : T_PPR_pat <S2_asr_r_p_or,   int_hexagon_S2_asr_r_p_or>;
def : T_PPR_pat <S2_lsr_r_p_or,   int_hexagon_S2_lsr_r_p_or>;
def : T_PPR_pat <S2_asl_r_p_or,   int_hexagon_S2_asl_r_p_or>;
def : T_PPR_pat <S2_lsl_r_p_or,   int_hexagon_S2_lsl_r_p_or>;

def : T_RRI_pat <S2_asr_i_r_nac,  int_hexagon_S2_asr_i_r_nac>;
def : T_RRI_pat <S2_lsr_i_r_nac,  int_hexagon_S2_lsr_i_r_nac>;
def : T_RRI_pat <S2_asl_i_r_nac,  int_hexagon_S2_asl_i_r_nac>;
def : T_RRI_pat <S2_asr_i_r_acc,  int_hexagon_S2_asr_i_r_acc>;
def : T_RRI_pat <S2_lsr_i_r_acc,  int_hexagon_S2_lsr_i_r_acc>;
def : T_RRI_pat <S2_asl_i_r_acc,  int_hexagon_S2_asl_i_r_acc>;

def : T_RRI_pat <S2_asr_i_r_and,  int_hexagon_S2_asr_i_r_and>;
def : T_RRI_pat <S2_lsr_i_r_and,  int_hexagon_S2_lsr_i_r_and>;
def : T_RRI_pat <S2_asl_i_r_and,  int_hexagon_S2_asl_i_r_and>;
def : T_RRI_pat <S2_asr_i_r_or,   int_hexagon_S2_asr_i_r_or>;
def : T_RRI_pat <S2_lsr_i_r_or,   int_hexagon_S2_lsr_i_r_or>;
def : T_RRI_pat <S2_asl_i_r_or,   int_hexagon_S2_asl_i_r_or>;
def : T_RRI_pat <S2_lsr_i_r_xacc, int_hexagon_S2_lsr_i_r_xacc>;
def : T_RRI_pat <S2_asl_i_r_xacc, int_hexagon_S2_asl_i_r_xacc>;

def : T_PPI_pat <S2_asr_i_p_nac,  int_hexagon_S2_asr_i_p_nac>;
def : T_PPI_pat <S2_lsr_i_p_nac,  int_hexagon_S2_lsr_i_p_nac>;
def : T_PPI_pat <S2_asl_i_p_nac,  int_hexagon_S2_asl_i_p_nac>;
def : T_PPI_pat <S2_asr_i_p_acc,  int_hexagon_S2_asr_i_p_acc>;
def : T_PPI_pat <S2_lsr_i_p_acc,  int_hexagon_S2_lsr_i_p_acc>;
def : T_PPI_pat <S2_asl_i_p_acc,  int_hexagon_S2_asl_i_p_acc>;

def : T_PPI_pat <S2_asr_i_p_and,  int_hexagon_S2_asr_i_p_and>;
def : T_PPI_pat <S2_lsr_i_p_and,  int_hexagon_S2_lsr_i_p_and>;
def : T_PPI_pat <S2_asl_i_p_and,  int_hexagon_S2_asl_i_p_and>;
def : T_PPI_pat <S2_asr_i_p_or,   int_hexagon_S2_asr_i_p_or>;
def : T_PPI_pat <S2_lsr_i_p_or,   int_hexagon_S2_lsr_i_p_or>;
def : T_PPI_pat <S2_asl_i_p_or,   int_hexagon_S2_asl_i_p_or>;
def : T_PPI_pat <S2_lsr_i_p_xacc, int_hexagon_S2_lsr_i_p_xacc>;
def : T_PPI_pat <S2_asl_i_p_xacc, int_hexagon_S2_asl_i_p_xacc>;

def : T_RRR_pat <S2_asr_r_r_nac,  int_hexagon_S2_asr_r_r_nac>;
def : T_RRR_pat <S2_lsr_r_r_nac,  int_hexagon_S2_lsr_r_r_nac>;
def : T_RRR_pat <S2_asl_r_r_nac,  int_hexagon_S2_asl_r_r_nac>;
def : T_RRR_pat <S2_lsl_r_r_nac,  int_hexagon_S2_lsl_r_r_nac>;
def : T_RRR_pat <S2_asr_r_r_acc,  int_hexagon_S2_asr_r_r_acc>;
def : T_RRR_pat <S2_lsr_r_r_acc,  int_hexagon_S2_lsr_r_r_acc>;
def : T_RRR_pat <S2_asl_r_r_acc,  int_hexagon_S2_asl_r_r_acc>;
def : T_RRR_pat <S2_lsl_r_r_acc,  int_hexagon_S2_lsl_r_r_acc>;

def : T_RRR_pat <S2_asr_r_r_and,  int_hexagon_S2_asr_r_r_and>;
def : T_RRR_pat <S2_lsr_r_r_and,  int_hexagon_S2_lsr_r_r_and>;
def : T_RRR_pat <S2_asl_r_r_and,  int_hexagon_S2_asl_r_r_and>;
def : T_RRR_pat <S2_lsl_r_r_and,  int_hexagon_S2_lsl_r_r_and>;
def : T_RRR_pat <S2_asr_r_r_or,   int_hexagon_S2_asr_r_r_or>;
def : T_RRR_pat <S2_lsr_r_r_or,   int_hexagon_S2_lsr_r_r_or>;
def : T_RRR_pat <S2_asl_r_r_or,   int_hexagon_S2_asl_r_r_or>;
def : T_RRR_pat <S2_lsl_r_r_or,   int_hexagon_S2_lsl_r_r_or>;

def : T_PPR_pat <S2_asr_r_p_nac,  int_hexagon_S2_asr_r_p_nac>;
def : T_PPR_pat <S2_lsr_r_p_nac,  int_hexagon_S2_lsr_r_p_nac>;
def : T_PPR_pat <S2_asl_r_p_nac,  int_hexagon_S2_asl_r_p_nac>;
def : T_PPR_pat <S2_lsl_r_p_nac,  int_hexagon_S2_lsl_r_p_nac>;
def : T_PPR_pat <S2_asr_r_p_acc,  int_hexagon_S2_asr_r_p_acc>;
def : T_PPR_pat <S2_lsr_r_p_acc,  int_hexagon_S2_lsr_r_p_acc>;
def : T_PPR_pat <S2_asl_r_p_acc,  int_hexagon_S2_asl_r_p_acc>;
def : T_PPR_pat <S2_lsl_r_p_acc,  int_hexagon_S2_lsl_r_p_acc>;

def : T_PPR_pat <S2_asr_r_p_and,  int_hexagon_S2_asr_r_p_and>;
def : T_PPR_pat <S2_lsr_r_p_and,  int_hexagon_S2_lsr_r_p_and>;
def : T_PPR_pat <S2_asl_r_p_and,  int_hexagon_S2_asl_r_p_and>;
def : T_PPR_pat <S2_lsl_r_p_and,  int_hexagon_S2_lsl_r_p_and>;
def : T_PPR_pat <S2_asr_r_p_or,   int_hexagon_S2_asr_r_p_or>;
def : T_PPR_pat <S2_lsr_r_p_or,   int_hexagon_S2_lsr_r_p_or>;
def : T_PPR_pat <S2_asl_r_p_or,   int_hexagon_S2_asl_r_p_or>;
def : T_PPR_pat <S2_lsl_r_p_or,   int_hexagon_S2_lsl_r_p_or>;

/********************************************************************
*            ALU32/ALU                                              *
*********************************************************************/
def : T_RR_pat<A2_add,      int_hexagon_A2_add>;
def : T_RI_pat<ADD_ri,      int_hexagon_A2_addi>;
def : T_RR_pat<A2_sub,      int_hexagon_A2_sub>;
def : T_IR_pat<SUB_ri,      int_hexagon_A2_subri>;
def : T_RR_pat<A2_and,      int_hexagon_A2_and>;
def : T_RI_pat<AND_ri,      int_hexagon_A2_andir>;
def : T_RR_pat<A2_or,       int_hexagon_A2_or>;
def : T_RI_pat<OR_ri,       int_hexagon_A2_orir>;
def : T_RR_pat<A2_xor,      int_hexagon_A2_xor>;
def : T_RR_pat<A2_combinew, int_hexagon_A2_combinew>;

// Assembler mapped from Rd32=not(Rs32) to Rd32=sub(#-1,Rs32)
def : Pat <(int_hexagon_A2_not (I32:$Rs)),
           (SUB_ri -1, IntRegs:$Rs)>;

// Assembler mapped from Rd32=neg(Rs32) to Rd32=sub(#0,Rs32)
def : Pat <(int_hexagon_A2_neg IntRegs:$Rs),
           (SUB_ri 0, IntRegs:$Rs)>;

// Transfer immediate
def  : Pat <(int_hexagon_A2_tfril (I32:$Rs), u16_0ImmPred:$Is),
            (A2_tfril IntRegs:$Rs, u16_0ImmPred:$Is)>;
def  : Pat <(int_hexagon_A2_tfrih (I32:$Rs), u16_0ImmPred:$Is),
            (A2_tfrih IntRegs:$Rs, u16_0ImmPred:$Is)>;

//  Transfer Register/immediate.
def : T_R_pat <A2_tfr, int_hexagon_A2_tfr>;
def : T_I_pat <A2_tfrsi, int_hexagon_A2_tfrsi>;

// Assembler mapped from Rdd32=Rss32 to Rdd32=combine(Rss.H32,Rss.L32)
def : Pat<(int_hexagon_A2_tfrp DoubleRegs:$src),
          (A2_combinew (HiReg DoubleRegs:$src), (LoReg DoubleRegs:$src))>;

/********************************************************************
*            ALU32/PERM                                             *
*********************************************************************/
// Combine
def: T_RR_pat<A2_combine_hh, int_hexagon_A2_combine_hh>;
def: T_RR_pat<A2_combine_hl, int_hexagon_A2_combine_hl>;
def: T_RR_pat<A2_combine_lh, int_hexagon_A2_combine_lh>;
def: T_RR_pat<A2_combine_ll, int_hexagon_A2_combine_ll>;

def: T_II_pat<A2_combineii, int_hexagon_A2_combineii, s8ExtPred, s8ImmPred>;

def: Pat<(i32 (int_hexagon_C2_mux (I32:$Rp), (I32:$Rs),
                                                     (I32:$Rt))),
         (i32 (C2_mux (C2_tfrrp IntRegs:$Rp), IntRegs:$Rs, IntRegs:$Rt))>;

// Mux
def : T_QRI_pat<C2_muxir, int_hexagon_C2_muxir, s8ExtPred>;
def : T_QIR_pat<C2_muxri, int_hexagon_C2_muxri, s8ExtPred>;
def : T_QII_pat<C2_muxii, int_hexagon_C2_muxii, s8ExtPred, s8ImmPred>;

// Shift halfword
def : T_R_pat<A2_aslh, int_hexagon_A2_aslh>;
def : T_R_pat<A2_asrh, int_hexagon_A2_asrh>;
def : T_R_pat<A2_asrh, int_hexagon_SI_to_SXTHI_asrh>;

// Sign/zero extend
def : T_R_pat<A2_sxth, int_hexagon_A2_sxth>;
def : T_R_pat<A2_sxtb, int_hexagon_A2_sxtb>;
def : T_R_pat<A2_zxth, int_hexagon_A2_zxth>;
def : T_R_pat<A2_zxtb, int_hexagon_A2_zxtb>;

/********************************************************************
*            ALU32/PRED                                             *
*********************************************************************/
// Compare
def : T_RR_pat<C2_cmpeq,  int_hexagon_C2_cmpeq>;
def : T_RR_pat<C2_cmpgt,  int_hexagon_C2_cmpgt>;
def : T_RR_pat<C2_cmpgtu, int_hexagon_C2_cmpgtu>;

def : T_RI_pat<C2_cmpeqi, int_hexagon_C2_cmpeqi, s10ExtPred>;
def : T_RI_pat<C2_cmpgti, int_hexagon_C2_cmpgti, s10ExtPred>;
def : T_RI_pat<C2_cmpgtui, int_hexagon_C2_cmpgtui, u9ExtPred>;

def : Pat <(i32 (int_hexagon_C2_cmpgei (I32:$src1), s8ExtPred:$src2)),
      (i32 (C2_cmpgti (I32:$src1),
                      (DEC_CONST_SIGNED s8ExtPred:$src2)))>;

def : Pat <(i32 (int_hexagon_C2_cmpgeui (I32:$src1), u8ExtPred:$src2)),
      (i32 (C2_cmpgtui (I32:$src1),
                       (DEC_CONST_UNSIGNED u8ExtPred:$src2)))>;

// The instruction, Pd=cmp.geu(Rs, #u8) -> Pd=cmp.eq(Rs,Rs) when #u8 == 0.
def : Pat <(i32 (int_hexagon_C2_cmpgeui (I32:$src1), 0)),
      (i32 (C2_cmpeq (I32:$src1), (I32:$src1)))>;

def : Pat <(i32 (int_hexagon_C2_cmplt (I32:$src1),
                                      (I32:$src2))),
      (i32 (C2_cmpgt (I32:$src2), (I32:$src1)))>;

def : Pat <(i32 (int_hexagon_C2_cmpltu (I32:$src1),
                                       (I32:$src2))),
      (i32 (C2_cmpgtu (I32:$src2), (I32:$src1)))>;

/********************************************************************
*            ALU64/ALU                                              *
*********************************************************************/
def: T_RR_pat<A2_addsat,   int_hexagon_A2_addsat>;
def: T_RR_pat<A2_subsat,   int_hexagon_A2_subsat>;
def: T_PP_pat<A2_addp,     int_hexagon_A2_addp>;
def: T_PP_pat<A2_subp,     int_hexagon_A2_subp>;

def: T_PP_pat<A2_andp,     int_hexagon_A2_andp>;
def: T_PP_pat<A2_orp,      int_hexagon_A2_orp>;
def: T_PP_pat<A2_xorp,     int_hexagon_A2_xorp>;

def: T_PP_pat<C2_cmpeqp,   int_hexagon_C2_cmpeqp>;
def: T_PP_pat<C2_cmpgtp,   int_hexagon_C2_cmpgtp>;
def: T_PP_pat<C2_cmpgtup,  int_hexagon_C2_cmpgtup>;

def: T_PP_pat<S2_parityp,  int_hexagon_S2_parityp>;
def: T_RR_pat<S2_packhl,   int_hexagon_S2_packhl>;

// MPY - Multiply and use full result
// Rdd = mpy[u](Rs, Rt)
def : T_RR_pat <M2_dpmpyss_s0, int_hexagon_M2_dpmpyss_s0>;
def : T_RR_pat <M2_dpmpyuu_s0, int_hexagon_M2_dpmpyuu_s0>;

// Rxx[+-]= mpy[u](Rs,Rt)
def : T_PRR_pat <M2_dpmpyss_acc_s0, int_hexagon_M2_dpmpyss_acc_s0>;
def : T_PRR_pat <M2_dpmpyss_nac_s0, int_hexagon_M2_dpmpyss_nac_s0>;
def : T_PRR_pat <M2_dpmpyuu_acc_s0, int_hexagon_M2_dpmpyuu_acc_s0>;
def : T_PRR_pat <M2_dpmpyuu_nac_s0, int_hexagon_M2_dpmpyuu_nac_s0>;

/********************************************************************
*            CR                                                     *
*********************************************************************/
class qi_CRInst_qi_pat<InstHexagon Inst, Intrinsic IntID> :
  Pat<(i32 (IntID IntRegs:$Rs)),
      (i32 (C2_tfrpr (Inst (C2_tfrrp IntRegs:$Rs))))>;

class qi_CRInst_qiqi_pat<InstHexagon Inst, Intrinsic IntID> :
  Pat<(i32 (IntID IntRegs:$Rs, IntRegs:$Rt)),
      (i32 (C2_tfrpr (Inst (C2_tfrrp IntRegs:$Rs), (C2_tfrrp IntRegs:$Rt))))>;

def: qi_CRInst_qi_pat<C2_not,     int_hexagon_C2_not>;
def: qi_CRInst_qi_pat<C2_all8,    int_hexagon_C2_all8>;
def: qi_CRInst_qi_pat<C2_any8,    int_hexagon_C2_any8>;

def: qi_CRInst_qiqi_pat<C2_and,   int_hexagon_C2_and>;
def: qi_CRInst_qiqi_pat<C2_andn,  int_hexagon_C2_andn>;
def: qi_CRInst_qiqi_pat<C2_or,    int_hexagon_C2_or>;
def: qi_CRInst_qiqi_pat<C2_orn,   int_hexagon_C2_orn>;
def: qi_CRInst_qiqi_pat<C2_xor,   int_hexagon_C2_xor>;

// Multiply 32x32 and use lower result
def : T_RRI_pat <M2_macsip, int_hexagon_M2_macsip>;
def : T_RRI_pat <M2_macsin, int_hexagon_M2_macsin>;
def : T_RRR_pat <M2_maci, int_hexagon_M2_maci>;

// Subtract and accumulate
def : T_RRR_pat <M2_subacc, int_hexagon_M2_subacc>;

// Add and accumulate
def : T_RRR_pat <M2_acci,   int_hexagon_M2_acci>;
def : T_RRR_pat <M2_nacci,  int_hexagon_M2_nacci>;
def : T_RRI_pat <M2_accii,  int_hexagon_M2_accii>;
def : T_RRI_pat <M2_naccii, int_hexagon_M2_naccii>;

// XOR and XOR with destination
def : T_RRR_pat <M2_xor_xacc, int_hexagon_M2_xor_xacc>;

class MType_R32_pat <Intrinsic IntID, InstHexagon OutputInst> :
      Pat <(IntID IntRegs:$src1, IntRegs:$src2),
           (OutputInst IntRegs:$src1, IntRegs:$src2)>;

// Multiply and use lower result
def : MType_R32_pat <int_hexagon_M2_mpyi, M2_mpyi>;
def : T_RI_pat<M2_mpysmi, int_hexagon_M2_mpysmi>;

// Assembler mapped from Rd32=mpyui(Rs32,Rt32) to Rd32=mpyi(Rs32,Rt32)
def : MType_R32_pat <int_hexagon_M2_mpyui, M2_mpyi>;

// Multiply and use upper result
def : MType_R32_pat <int_hexagon_M2_mpy_up, M2_mpy_up>;
def : MType_R32_pat <int_hexagon_M2_mpyu_up, M2_mpyu_up>;
def : MType_R32_pat <int_hexagon_M2_hmmpyh_rs1, M2_hmmpyh_rs1>;
def : MType_R32_pat <int_hexagon_M2_hmmpyl_rs1, M2_hmmpyl_rs1>;
def : MType_R32_pat <int_hexagon_M2_dpmpyss_rnd_s0, M2_dpmpyss_rnd_s0>;

/********************************************************************
*            STYPE/ALU                                              *
*********************************************************************/
def : T_P_pat <A2_absp, int_hexagon_A2_absp>;
def : T_P_pat <A2_negp, int_hexagon_A2_negp>;
def : T_P_pat <A2_notp, int_hexagon_A2_notp>;

/********************************************************************
*            STYPE/BIT                                              *
*********************************************************************/

// Count leading/trailing
def: T_R_pat<S2_cl0,     int_hexagon_S2_cl0>;
def: T_P_pat<S2_cl0p,    int_hexagon_S2_cl0p>;
def: T_R_pat<S2_cl1,     int_hexagon_S2_cl1>;
def: T_P_pat<S2_cl1p,    int_hexagon_S2_cl1p>;
def: T_R_pat<S2_clb,     int_hexagon_S2_clb>;
def: T_P_pat<S2_clbp,    int_hexagon_S2_clbp>;
def: T_R_pat<S2_clbnorm, int_hexagon_S2_clbnorm>;
def: T_R_pat<S2_ct0,     int_hexagon_S2_ct0>;
def: T_R_pat<S2_ct1,     int_hexagon_S2_ct1>;

// Compare bit mask
def: T_RR_pat<C2_bitsclr,  int_hexagon_C2_bitsclr>;
def: T_RI_pat<C2_bitsclri, int_hexagon_C2_bitsclri>;
def: T_RR_pat<C2_bitsset,  int_hexagon_C2_bitsset>;

// Linear feedback-shift Iteration.
def : T_PP_pat <S2_lfsp, int_hexagon_S2_lfsp>;

// Shift by immediate and add
def : T_RRI_pat<S2_addasl_rrri, int_hexagon_S2_addasl_rrri>;

// Extract bitfield
def : T_PII_pat<S2_extractup,    int_hexagon_S2_extractup>;
def : T_RII_pat<S2_extractu,     int_hexagon_S2_extractu>;
def : T_RP_pat <S2_extractu_rp,  int_hexagon_S2_extractu_rp>;
def : T_PP_pat <S2_extractup_rp, int_hexagon_S2_extractup_rp>;

// Insert bitfield
def : Pat <(int_hexagon_S2_insert_rp IntRegs:$src1, IntRegs:$src2,
                                     DoubleRegs:$src3),
           (S2_insert_rp IntRegs:$src1, IntRegs:$src2, DoubleRegs:$src3)>;

def : Pat<(i64 (int_hexagon_S2_insertp_rp (I64:$src1),
                 (I64:$src2), (I64:$src3))),
          (i64 (S2_insertp_rp (I64:$src1), (I64:$src2),
                              (I64:$src3)))>;

def : Pat<(int_hexagon_S2_insert IntRegs:$src1, IntRegs:$src2,
                                 u5ImmPred:$src3, u5ImmPred:$src4),
          (S2_insert IntRegs:$src1, IntRegs:$src2,
                     u5ImmPred:$src3, u5ImmPred:$src4)>;

def : Pat<(i64 (int_hexagon_S2_insertp (I64:$src1),
                 (I64:$src2), u6ImmPred:$src3, u6ImmPred:$src4)),
          (i64 (S2_insertp (I64:$src1), (I64:$src2),
                           u6ImmPred:$src3, u6ImmPred:$src4))>;


// Innterleave/deinterleave
def : T_P_pat <S2_interleave, int_hexagon_S2_interleave>;
def : T_P_pat <S2_deinterleave, int_hexagon_S2_deinterleave>;

// Set/Clear/Toggle Bit
def: T_RI_pat<S2_setbit_i,    int_hexagon_S2_setbit_i>;
def: T_RI_pat<S2_clrbit_i,    int_hexagon_S2_clrbit_i>;
def: T_RI_pat<S2_togglebit_i, int_hexagon_S2_togglebit_i>;

def: T_RR_pat<S2_setbit_r,    int_hexagon_S2_setbit_r>;
def: T_RR_pat<S2_clrbit_r,    int_hexagon_S2_clrbit_r>;
def: T_RR_pat<S2_togglebit_r, int_hexagon_S2_togglebit_r>;

// Test Bit
def: T_RI_pat<S2_tstbit_i,    int_hexagon_S2_tstbit_i>;
def: T_RR_pat<S2_tstbit_r,    int_hexagon_S2_tstbit_r>;

/********************************************************************
*            STYPE/SHIFT                                            *
*********************************************************************/

def : T_PI_pat <S2_asr_i_p, int_hexagon_S2_asr_i_p>;
def : T_PI_pat <S2_lsr_i_p, int_hexagon_S2_lsr_i_p>;
def : T_PI_pat <S2_asl_i_p, int_hexagon_S2_asl_i_p>;

def : T_PR_pat <S2_asr_r_p, int_hexagon_S2_asr_r_p>;
def : T_PR_pat <S2_lsr_r_p, int_hexagon_S2_lsr_r_p>;
def : T_PR_pat <S2_asl_r_p, int_hexagon_S2_asl_r_p>;
def : T_PR_pat <S2_lsl_r_p, int_hexagon_S2_lsl_r_p>;

def : T_RR_pat <S2_asr_r_r, int_hexagon_S2_asr_r_r>;
def : T_RR_pat <S2_lsr_r_r, int_hexagon_S2_lsr_r_r>;
def : T_RR_pat <S2_asl_r_r, int_hexagon_S2_asl_r_r>;
def : T_RR_pat <S2_lsl_r_r, int_hexagon_S2_lsl_r_r>;

def : T_RR_pat <S2_asr_r_r_sat, int_hexagon_S2_asr_r_r_sat>;
def : T_RR_pat <S2_asl_r_r_sat, int_hexagon_S2_asl_r_r_sat>;

def : T_R_pat <A2_sxtw,     int_hexagon_A2_sxtw>;

def : T_R_pat <S2_brev,        int_hexagon_S2_brev>;

def : T_R_pat <A2_abs,    int_hexagon_A2_abs>;
def : T_R_pat <A2_abssat, int_hexagon_A2_abssat>;
def : T_R_pat <A2_negsat, int_hexagon_A2_negsat>;

def : T_R_pat <A2_swiz,   int_hexagon_A2_swiz>;

def : T_P_pat <A2_sat,    int_hexagon_A2_sat>;
def : T_R_pat <A2_sath,   int_hexagon_A2_sath>;
def : T_R_pat <A2_satuh,  int_hexagon_A2_satuh>;
def : T_R_pat <A2_satub,  int_hexagon_A2_satub>;
def : T_R_pat <A2_satb,   int_hexagon_A2_satb>;

def : T_RI_pat <S2_asr_i_r,     int_hexagon_S2_asr_i_r>;
def : T_RI_pat <S2_lsr_i_r,     int_hexagon_S2_lsr_i_r>;
def : T_RI_pat <S2_asl_i_r,     int_hexagon_S2_asl_i_r>;
def : T_RI_pat <S2_asr_i_r_rnd, int_hexagon_S2_asr_i_r_rnd>;
def : T_RI_pat <S2_asr_i_r_rnd_goodsyntax,
                int_hexagon_S2_asr_i_r_rnd_goodsyntax>;

// Shift left by immediate with saturation.
def : T_RI_pat <S2_asl_i_r_sat, int_hexagon_S2_asl_i_r_sat>;

//===----------------------------------------------------------------------===//
// Template 'def pat' to map tableidx[bhwd] intrinsics to :raw instructions.
//===----------------------------------------------------------------------===//
class S2op_tableidx_pat <Intrinsic IntID, InstHexagon OutputInst,
                         SDNodeXForm XformImm>
  : Pat <(IntID IntRegs:$src1, IntRegs:$src2, u4ImmPred:$src3, u5ImmPred:$src4),
         (OutputInst IntRegs:$src1, IntRegs:$src2, u4ImmPred:$src3,
                     (XformImm u5ImmPred:$src4))>;


// Table Index : Extract and insert bits.
// Map to the real hardware instructions after subtracting appropriate
// values from the 4th input operand. Please note that subtraction is not
// needed for int_hexagon_S2_tableidxb_goodsyntax.

def : Pat <(int_hexagon_S2_tableidxb_goodsyntax IntRegs:$src1, IntRegs:$src2,
                                              u4ImmPred:$src3, u5ImmPred:$src4),
           (S2_tableidxb IntRegs:$src1, IntRegs:$src2,
                         u4ImmPred:$src3, u5ImmPred:$src4)>;

def : S2op_tableidx_pat <int_hexagon_S2_tableidxh_goodsyntax, S2_tableidxh,
                         DEC_CONST_SIGNED>;
def : S2op_tableidx_pat <int_hexagon_S2_tableidxw_goodsyntax, S2_tableidxw,
                         DEC2_CONST_SIGNED>;
def : S2op_tableidx_pat <int_hexagon_S2_tableidxd_goodsyntax, S2_tableidxd,
                         DEC3_CONST_SIGNED>;

//
// ALU 32 types.
//

class qi_ALU32_sisi<string opc, Intrinsic IntID>
  : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class qi_ALU32_sis10<string opc, Intrinsic IntID>
  : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$src1, s10Imm:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class qi_ALU32_sis8<string opc, Intrinsic IntID>
  : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$src1, s8Imm:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class qi_ALU32_siu8<string opc, Intrinsic IntID>
  : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$src1, u8Imm:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class qi_ALU32_siu9<string opc, Intrinsic IntID>
  : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$src1, u9Imm:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class si_ALU32_qisisi<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2,
                                      IntRegs:$src3),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2, $src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2,
                                        IntRegs:$src3))]>;

class si_ALU32_sisi<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU32_sisi_sat<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU32_sisi_rnd<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):rnd")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_ALU64_di<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
             !strconcat("$dst = ", !strconcat(opc , "$src")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src))]>;

//
// ALU 64 types.
//

class di_ALU64_didi<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_ALU64_qididi<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src1, DoubleRegs:$src2,
                                          DoubleRegs:$src3),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2, $src3)")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, DoubleRegs:$src2,
                                           DoubleRegs:$src3))]>;

class di_ALU64_didi_sat<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_ALU64_didi_rnd<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):rnd")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_ALU64_didi_crnd<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):crnd")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_ALU64_didi_rnd_sat<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):rnd:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_ALU64_didi_crnd_sat<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):crnd:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class qi_ALU64_didi<string opc, Intrinsic IntID>
  : ALU64_rr<(outs PredRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set PredRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2))]>;

//
// SInst classes.
//

class qi_SInst_qi<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src),
             !strconcat("$dst = ", !strconcat(opc , "($src)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src))]>;

class qi_SInst_qi_pxfer<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src),
             !strconcat("$dst = ", !strconcat(opc , "$src")),
             [(set PredRegs:$dst, (IntID IntRegs:$src))]>;

class qi_SInst_qiqi<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class qi_SInst_qiqi_neg<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, !$src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_SInst_di<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
             !strconcat("$dst = ", !strconcat(opc , "($src)")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src))]>;

class di_SInst_di_sat<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
             !strconcat("$dst = ", !strconcat(opc , "($src):sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src))]>;

class si_SInst_di<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins DoubleRegs:$src),
          !strconcat("$dst = ", !strconcat(opc , "($src)")),
          [(set IntRegs:$dst, (IntID DoubleRegs:$src))]>;

class si_SInst_di_sat<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins DoubleRegs:$src),
          !strconcat("$dst = ", !strconcat(opc , "($src):sat")),
          [(set IntRegs:$dst, (IntID DoubleRegs:$src))]>;

class di_SInst_disi<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, IntRegs:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, IntRegs:$src2))]>;

class di_SInst_didi<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2))]>;

class di_SInst_si<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1),
          !strconcat("$dst = ", !strconcat(opc , "($src1)")),
          [(set DoubleRegs:$dst, (IntID IntRegs:$src1))]>;

class si_SInst_diu5<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, u5Imm:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
          [(set IntRegs:$dst, (IntID DoubleRegs:$src1, imm:$src2))]>;

class si_SInst_disi<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, IntRegs:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
          [(set IntRegs:$dst, (IntID DoubleRegs:$src1, IntRegs:$src2))]>;

class si_SInst_si<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src),
          !strconcat("$dst = ", !strconcat(opc , "($src)")),
          [(set IntRegs:$dst, (IntID IntRegs:$src))]>;

class di_SInst_qi<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins IntRegs:$src),
          !strconcat("$dst = ", !strconcat(opc , "($src)")),
          [(set DoubleRegs:$dst, (IntID IntRegs:$src))]>;

class si_SInst_qi<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src),
          !strconcat("$dst = ", !strconcat(opc , "$src")),
          [(set IntRegs:$dst, (IntID IntRegs:$src))]>;

class si_SInst_qiqi<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
          [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class qi_SInst_si<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src),
          !strconcat("$dst = ", !strconcat(opc , "$src")),
          [(set PredRegs:$dst, (IntID IntRegs:$src))]>;

class di_SInst_didiqi<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2,
                                       IntRegs:$src3),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2, $src3)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2,
                                        IntRegs:$src3))]>;

class di_SInst_didiu3<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2,
                                       u3Imm:$src3),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2, #$src3)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2,
                                        imm:$src3))]>;


//
// MInst classes.
//

class di_MInst_disisi_acc<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst += ", !strconcat(opc , "($src1, $src2)")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_disisi_nac<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst -= ", !strconcat(opc , "($src1, $src2)")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_disisi_acc_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst += ", !strconcat(opc , "($src1, $src2):sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_disisi_nac_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst -= ", !strconcat(opc , "($src1, $src2):sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_disisi_acc_sat_conj<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst += ", !strconcat(opc , "($src1, $src2*):sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_disisi_nac_sat_conj<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst -= ", !strconcat(opc , "($src1, $src2*):sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_disisi_nac_s1_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst -= ", !strconcat(opc ,
                                               "($src1, $src2):<<1:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_disisi_acc_s1_sat_conj<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst += ", !strconcat(opc ,
                                               "($src1, $src2*):<<1:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_disisi_nac_s1_sat_conj<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst -= ", !strconcat(opc ,
                                               "($src1, $src2*):<<1:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_didi<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_MInst_didi_conj<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2*)")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_MInst_sisi_s1_sat_conj<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2*):<<1:sat")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_didi_s1_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2):<<1:rnd:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_MInst_didi_sat<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_MInst_didi_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2):rnd:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class si_SInst_didi_sat<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):sat")),
          [(set IntRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2))]>;

class si_SInst_disi_s1_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2):<<1:rnd:sat")),
             [(set IntRegs:$dst, (IntID DoubleRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_s1_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2):<<1:rnd:sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_rnd_sat_conj<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2*):rnd:sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_s1_rnd_sat_conj<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2*):<<1:rnd:sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2):rnd:sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_sat<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):sat")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_sat_conj<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2*):sat")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_s1_sat<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):<<1:sat")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_didi_s1_sat<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):<<1:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class si_MInst_didi_s1_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2):<<1:rnd:sat")),
             [(set IntRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2))]>;

class si_MInst_didi_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):rnd:sat")),
             [(set IntRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2))]>;

class di_MInst_dididi_acc_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2,
                                           DoubleRegs:$src1, DoubleRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc , "($src1, $src2):sat")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                             DoubleRegs:$src1,
                                             DoubleRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_dididi_acc_rnd_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           DoubleRegs:$src2),
               !strconcat("$dst += ",
                          !strconcat(opc , "($src1, $src2):rnd:sat")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                             DoubleRegs:$src1,
                                             DoubleRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_dididi_acc_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2,
                                           DoubleRegs:$src1,
                                           DoubleRegs:$src2),
               !strconcat("$dst += ",
                          !strconcat(opc , "($src1, $src2):<<1")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                             DoubleRegs:$src1,
                                             DoubleRegs:$src2))],
               "$dst2 = $dst">;


class di_MInst_dididi_acc_s1_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2,
                                           DoubleRegs:$src1,
                                           DoubleRegs:$src2),
               !strconcat("$dst += ",
                          !strconcat(opc , "($src1, $src2):<<1:sat")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                             DoubleRegs:$src1,
                                             DoubleRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_dididi_acc_s1_rnd_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           DoubleRegs:$src2),
               !strconcat("$dst += ",
                          !strconcat(opc , "($src1, $src2):<<1:rnd:sat")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                             DoubleRegs:$src1,
                                             DoubleRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_dididi_acc<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           DoubleRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc , "($src1, $src2)")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                             DoubleRegs:$src1,
                                             DoubleRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_dididi_acc_conj<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           DoubleRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc , "($src1, $src2*)")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                             DoubleRegs:$src1,
                                             DoubleRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_acc_s1_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst += ",
                          !strconcat(opc , "($src1, $src2):<<1:sat")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disi_s1_sat<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):<<1:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, IntRegs:$src2))]>;

class di_MInst_didisi_acc_s1_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst += ",
                          !strconcat(opc , "($src1, $src2):<<1:sat")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                             DoubleRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_disi_s1_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ",
                        !strconcat(opc , "($src1, $src2):<<1:rnd:sat")),
             [(set IntRegs:$dst, (IntID DoubleRegs:$src1, IntRegs:$src2))]>;

class si_MInst_didi<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set IntRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2))]>;

//
// LDInst classes.
//
let mayLoad = 1, hasSideEffects = 0 in
class di_LDInstPI_diu4<string opc, Intrinsic IntID>
  : LDInstPI<(outs IntRegs:$dst, DoubleRegs:$dst2),
           (ins IntRegs:$src1, IntRegs:$src2, CtrRegs:$src3, s4Imm:$offset),
           "$dst2 = memd($src1++#$offset:circ($src3))",
           [],
           "$src1 = $dst">;

/********************************************************************
*            ALU32/VH                                               *
*********************************************************************/

// ALU32 / VH / Vector add halfwords.
// Rd32=vadd[u]h(Rs32,Rt32:sat]
def HEXAGON_A2_svaddh:
  si_ALU32_sisi                   <"vaddh",    int_hexagon_A2_svaddh>;
def HEXAGON_A2_svaddhs:
  si_ALU32_sisi_sat               <"vaddh",    int_hexagon_A2_svaddhs>;
def HEXAGON_A2_svadduhs:
  si_ALU32_sisi_sat               <"vadduh",   int_hexagon_A2_svadduhs>;

// ALU32 / VH / Vector average halfwords.
def HEXAGON_A2_svavgh:
  si_ALU32_sisi                   <"vavgh",    int_hexagon_A2_svavgh>;
def HEXAGON_A2_svavghs:
  si_ALU32_sisi_rnd               <"vavgh",    int_hexagon_A2_svavghs>;
def HEXAGON_A2_svnavgh:
  si_ALU32_sisi                   <"vnavgh",   int_hexagon_A2_svnavgh>;

// ALU32 / VH / Vector subtract halfwords.
def HEXAGON_A2_svsubh:
  si_ALU32_sisi                   <"vsubh",    int_hexagon_A2_svsubh>;
def HEXAGON_A2_svsubhs:
  si_ALU32_sisi_sat               <"vsubh",    int_hexagon_A2_svsubhs>;
def HEXAGON_A2_svsubuhs:
  si_ALU32_sisi_sat               <"vsubuh",   int_hexagon_A2_svsubuhs>;

// ALU64 / ALU / Transfer register.
def HEXAGON_A2_tfrp:
  di_ALU64_di                     <"",         int_hexagon_A2_tfrp>;

/********************************************************************
*            ALU64/VB                                               *
*********************************************************************/

// ALU64 / VB / Vector add unsigned bytes.
def HEXAGON_A2_vaddub:
  di_ALU64_didi                   <"vaddub",   int_hexagon_A2_vaddub>;
def HEXAGON_A2_vaddubs:
  di_ALU64_didi_sat               <"vaddub",   int_hexagon_A2_vaddubs>;

// ALU64 / VB / Vector average unsigned bytes.
def HEXAGON_A2_vavgub:
  di_ALU64_didi                   <"vavgub",   int_hexagon_A2_vavgub>;
def HEXAGON_A2_vavgubr:
  di_ALU64_didi_rnd               <"vavgub",   int_hexagon_A2_vavgubr>;

// ALU64 / VB / Vector compare unsigned bytes.
def HEXAGON_A2_vcmpbeq:
  qi_ALU64_didi                   <"vcmpb.eq", int_hexagon_A2_vcmpbeq>;
def HEXAGON_A2_vcmpbgtu:
  qi_ALU64_didi                   <"vcmpb.gtu",int_hexagon_A2_vcmpbgtu>;

// ALU64 / VB / Vector maximum/minimum unsigned bytes.
def HEXAGON_A2_vmaxub:
  di_ALU64_didi                   <"vmaxub",   int_hexagon_A2_vmaxub>;
def HEXAGON_A2_vminub:
  di_ALU64_didi                   <"vminub",   int_hexagon_A2_vminub>;

// ALU64 / VB / Vector subtract unsigned bytes.
def HEXAGON_A2_vsubub:
  di_ALU64_didi                   <"vsubub",   int_hexagon_A2_vsubub>;
def HEXAGON_A2_vsububs:
  di_ALU64_didi_sat               <"vsubub",   int_hexagon_A2_vsububs>;

// ALU64 / VB / Vector mux.
def HEXAGON_C2_vmux:
  di_ALU64_qididi                 <"vmux",     int_hexagon_C2_vmux>;


/********************************************************************
*            ALU64/VH                                               *
*********************************************************************/

// ALU64 / VH / Vector add halfwords.
// Rdd64=vadd[u]h(Rss64,Rtt64:sat]
def HEXAGON_A2_vaddh:
  di_ALU64_didi                   <"vaddh",    int_hexagon_A2_vaddh>;
def HEXAGON_A2_vaddhs:
  di_ALU64_didi_sat               <"vaddh",    int_hexagon_A2_vaddhs>;
def HEXAGON_A2_vadduhs:
  di_ALU64_didi_sat               <"vadduh",   int_hexagon_A2_vadduhs>;

// ALU64 / VH / Vector average halfwords.
// Rdd64=v[n]avg[u]h(Rss64,Rtt64:rnd/:crnd][:sat]
def HEXAGON_A2_vavgh:
  di_ALU64_didi                   <"vavgh",    int_hexagon_A2_vavgh>;
def HEXAGON_A2_vavghcr:
  di_ALU64_didi_crnd              <"vavgh",    int_hexagon_A2_vavghcr>;
def HEXAGON_A2_vavghr:
  di_ALU64_didi_rnd               <"vavgh",    int_hexagon_A2_vavghr>;
def HEXAGON_A2_vavguh:
  di_ALU64_didi                   <"vavguh",   int_hexagon_A2_vavguh>;
def HEXAGON_A2_vavguhr:
  di_ALU64_didi_rnd               <"vavguh",   int_hexagon_A2_vavguhr>;
def HEXAGON_A2_vnavgh:
  di_ALU64_didi                   <"vnavgh",   int_hexagon_A2_vnavgh>;
def HEXAGON_A2_vnavghcr:
  di_ALU64_didi_crnd_sat          <"vnavgh",   int_hexagon_A2_vnavghcr>;
def HEXAGON_A2_vnavghr:
  di_ALU64_didi_rnd_sat           <"vnavgh",   int_hexagon_A2_vnavghr>;

// ALU64 / VH / Vector compare halfwords.
def HEXAGON_A2_vcmpheq:
  qi_ALU64_didi                   <"vcmph.eq", int_hexagon_A2_vcmpheq>;
def HEXAGON_A2_vcmphgt:
  qi_ALU64_didi                   <"vcmph.gt", int_hexagon_A2_vcmphgt>;
def HEXAGON_A2_vcmphgtu:
  qi_ALU64_didi                   <"vcmph.gtu",int_hexagon_A2_vcmphgtu>;

// ALU64 / VH / Vector maximum halfwords.
def HEXAGON_A2_vmaxh:
  di_ALU64_didi                   <"vmaxh",    int_hexagon_A2_vmaxh>;
def HEXAGON_A2_vmaxuh:
  di_ALU64_didi                   <"vmaxuh",   int_hexagon_A2_vmaxuh>;

// ALU64 / VH / Vector minimum halfwords.
def HEXAGON_A2_vminh:
  di_ALU64_didi                   <"vminh",    int_hexagon_A2_vminh>;
def HEXAGON_A2_vminuh:
  di_ALU64_didi                   <"vminuh",   int_hexagon_A2_vminuh>;

// ALU64 / VH / Vector subtract halfwords.
def HEXAGON_A2_vsubh:
  di_ALU64_didi                   <"vsubh",    int_hexagon_A2_vsubh>;
def HEXAGON_A2_vsubhs:
  di_ALU64_didi_sat               <"vsubh",    int_hexagon_A2_vsubhs>;
def HEXAGON_A2_vsubuhs:
  di_ALU64_didi_sat               <"vsubuh",   int_hexagon_A2_vsubuhs>;


/********************************************************************
*            ALU64/VW                                               *
*********************************************************************/

// ALU64 / VW / Vector add words.
// Rdd32=vaddw(Rss32,Rtt32)[:sat]
def HEXAGON_A2_vaddw:
  di_ALU64_didi                   <"vaddw",    int_hexagon_A2_vaddw>;
def HEXAGON_A2_vaddws:
  di_ALU64_didi_sat               <"vaddw",   int_hexagon_A2_vaddws>;

// ALU64 / VW / Vector average words.
def HEXAGON_A2_vavguw:
  di_ALU64_didi                   <"vavguw",   int_hexagon_A2_vavguw>;
def HEXAGON_A2_vavguwr:
  di_ALU64_didi_rnd               <"vavguw",   int_hexagon_A2_vavguwr>;
def HEXAGON_A2_vavgw:
  di_ALU64_didi                   <"vavgw",    int_hexagon_A2_vavgw>;
def HEXAGON_A2_vavgwcr:
  di_ALU64_didi_crnd              <"vavgw",    int_hexagon_A2_vavgwcr>;
def HEXAGON_A2_vavgwr:
  di_ALU64_didi_rnd               <"vavgw",    int_hexagon_A2_vavgwr>;
def HEXAGON_A2_vnavgw:
  di_ALU64_didi                   <"vnavgw",   int_hexagon_A2_vnavgw>;
def HEXAGON_A2_vnavgwcr:
  di_ALU64_didi_crnd_sat          <"vnavgw",   int_hexagon_A2_vnavgwcr>;
def HEXAGON_A2_vnavgwr:
  di_ALU64_didi_rnd_sat           <"vnavgw",   int_hexagon_A2_vnavgwr>;

// ALU64 / VW / Vector compare words.
def HEXAGON_A2_vcmpweq:
  qi_ALU64_didi                   <"vcmpw.eq", int_hexagon_A2_vcmpweq>;
def HEXAGON_A2_vcmpwgt:
  qi_ALU64_didi                   <"vcmpw.gt", int_hexagon_A2_vcmpwgt>;
def HEXAGON_A2_vcmpwgtu:
  qi_ALU64_didi                   <"vcmpw.gtu",int_hexagon_A2_vcmpwgtu>;

// ALU64 / VW / Vector maximum words.
def HEXAGON_A2_vmaxw:
  di_ALU64_didi                   <"vmaxw",    int_hexagon_A2_vmaxw>;
def HEXAGON_A2_vmaxuw:
  di_ALU64_didi                   <"vmaxuw",   int_hexagon_A2_vmaxuw>;

// ALU64 / VW / Vector minimum words.
def HEXAGON_A2_vminw:
  di_ALU64_didi                   <"vminw",    int_hexagon_A2_vminw>;
def HEXAGON_A2_vminuw:
  di_ALU64_didi                   <"vminuw",   int_hexagon_A2_vminuw>;

// ALU64 / VW / Vector subtract words.
def HEXAGON_A2_vsubw:
  di_ALU64_didi                   <"vsubw",    int_hexagon_A2_vsubw>;
def HEXAGON_A2_vsubws:
  di_ALU64_didi_sat               <"vsubw",    int_hexagon_A2_vsubws>;

/********************************************************************
*            MTYPE/ALU                                              *
*********************************************************************/

// MTYPE / ALU / Vector absolute difference.
def HEXAGON_M2_vabsdiffh:
  di_MInst_didi                   <"vabsdiffh",int_hexagon_M2_vabsdiffh>;
def HEXAGON_M2_vabsdiffw:
  di_MInst_didi                   <"vabsdiffw",int_hexagon_M2_vabsdiffw>;


/********************************************************************
*            MTYPE/COMPLEX                                          *
*********************************************************************/

// MTYPE / COMPLEX / Complex multiply.
// Rdd[-+]=cmpy(Rs, Rt:<<1]:sat
def HEXAGON_M2_cmpys_s1:
  di_MInst_sisi_s1_sat            <"cmpy",     int_hexagon_M2_cmpys_s1>;
def HEXAGON_M2_cmpys_s0:
  di_MInst_sisi_sat               <"cmpy",     int_hexagon_M2_cmpys_s0>;
def HEXAGON_M2_cmpysc_s1:
  di_MInst_sisi_s1_sat_conj       <"cmpy",     int_hexagon_M2_cmpysc_s1>;
def HEXAGON_M2_cmpysc_s0:
  di_MInst_sisi_sat_conj          <"cmpy",     int_hexagon_M2_cmpysc_s0>;

def HEXAGON_M2_cmacs_s1:
  di_MInst_disisi_acc_s1_sat      <"cmpy",     int_hexagon_M2_cmacs_s1>;
def HEXAGON_M2_cmacs_s0:
  di_MInst_disisi_acc_sat         <"cmpy",     int_hexagon_M2_cmacs_s0>;
def HEXAGON_M2_cmacsc_s1:
  di_MInst_disisi_acc_s1_sat_conj <"cmpy",     int_hexagon_M2_cmacsc_s1>;
def HEXAGON_M2_cmacsc_s0:
  di_MInst_disisi_acc_sat_conj    <"cmpy",     int_hexagon_M2_cmacsc_s0>;

def HEXAGON_M2_cnacs_s1:
  di_MInst_disisi_nac_s1_sat      <"cmpy",     int_hexagon_M2_cnacs_s1>;
def HEXAGON_M2_cnacs_s0:
  di_MInst_disisi_nac_sat         <"cmpy",     int_hexagon_M2_cnacs_s0>;
def HEXAGON_M2_cnacsc_s1:
  di_MInst_disisi_nac_s1_sat_conj <"cmpy",     int_hexagon_M2_cnacsc_s1>;
def HEXAGON_M2_cnacsc_s0:
  di_MInst_disisi_nac_sat_conj    <"cmpy",     int_hexagon_M2_cnacsc_s0>;

// MTYPE / COMPLEX / Complex multiply real or imaginary.
def HEXAGON_M2_cmpyr_s0:
  di_MInst_sisi                   <"cmpyr",    int_hexagon_M2_cmpyr_s0>;
def HEXAGON_M2_cmacr_s0:
  di_MInst_disisi_acc             <"cmpyr",    int_hexagon_M2_cmacr_s0>;

def HEXAGON_M2_cmpyi_s0:
  di_MInst_sisi                   <"cmpyi",    int_hexagon_M2_cmpyi_s0>;
def HEXAGON_M2_cmaci_s0:
  di_MInst_disisi_acc             <"cmpyi",    int_hexagon_M2_cmaci_s0>;

// MTYPE / COMPLEX / Complex multiply with round and pack.
// Rxx32+=cmpy(Rs32,[*]Rt32:<<1]:rnd:sat
def HEXAGON_M2_cmpyrs_s0:
  si_MInst_sisi_rnd_sat           <"cmpy",     int_hexagon_M2_cmpyrs_s0>;
def HEXAGON_M2_cmpyrs_s1:
  si_MInst_sisi_s1_rnd_sat        <"cmpy",     int_hexagon_M2_cmpyrs_s1>;

def HEXAGON_M2_cmpyrsc_s0:
  si_MInst_sisi_rnd_sat_conj      <"cmpy",     int_hexagon_M2_cmpyrsc_s0>;
def HEXAGON_M2_cmpyrsc_s1:
  si_MInst_sisi_s1_rnd_sat_conj   <"cmpy",     int_hexagon_M2_cmpyrsc_s1>;

//MTYPE / COMPLEX / Vector complex multiply real or imaginary.
def HEXAGON_M2_vcmpy_s0_sat_i:
  di_MInst_didi_sat               <"vcmpyi",   int_hexagon_M2_vcmpy_s0_sat_i>;
def HEXAGON_M2_vcmpy_s1_sat_i:
  di_MInst_didi_s1_sat            <"vcmpyi",   int_hexagon_M2_vcmpy_s1_sat_i>;

def HEXAGON_M2_vcmpy_s0_sat_r:
  di_MInst_didi_sat               <"vcmpyr",   int_hexagon_M2_vcmpy_s0_sat_r>;
def HEXAGON_M2_vcmpy_s1_sat_r:
  di_MInst_didi_s1_sat            <"vcmpyr",   int_hexagon_M2_vcmpy_s1_sat_r>;

def HEXAGON_M2_vcmac_s0_sat_i:
  di_MInst_dididi_acc_sat         <"vcmpyi",   int_hexagon_M2_vcmac_s0_sat_i>;
def HEXAGON_M2_vcmac_s0_sat_r:
  di_MInst_dididi_acc_sat         <"vcmpyr",   int_hexagon_M2_vcmac_s0_sat_r>;

//MTYPE / COMPLEX / Vector reduce complex multiply real or imaginary.
def HEXAGON_M2_vrcmpyi_s0:
  di_MInst_didi                   <"vrcmpyi",  int_hexagon_M2_vrcmpyi_s0>;
def HEXAGON_M2_vrcmpyr_s0:
  di_MInst_didi                   <"vrcmpyr",  int_hexagon_M2_vrcmpyr_s0>;

def HEXAGON_M2_vrcmpyi_s0c:
  di_MInst_didi_conj              <"vrcmpyi",  int_hexagon_M2_vrcmpyi_s0c>;
def HEXAGON_M2_vrcmpyr_s0c:
  di_MInst_didi_conj              <"vrcmpyr",  int_hexagon_M2_vrcmpyr_s0c>;

def HEXAGON_M2_vrcmaci_s0:
  di_MInst_dididi_acc             <"vrcmpyi",  int_hexagon_M2_vrcmaci_s0>;
def HEXAGON_M2_vrcmacr_s0:
  di_MInst_dididi_acc             <"vrcmpyr",  int_hexagon_M2_vrcmacr_s0>;

def HEXAGON_M2_vrcmaci_s0c:
  di_MInst_dididi_acc_conj        <"vrcmpyi",  int_hexagon_M2_vrcmaci_s0c>;
def HEXAGON_M2_vrcmacr_s0c:
  di_MInst_dididi_acc_conj        <"vrcmpyr",  int_hexagon_M2_vrcmacr_s0c>;


/********************************************************************
*            MTYPE/MPYH                                             *
*********************************************************************/

// MTYPE / MPYH / Multiply word by half (32x16).
//Rdd[+]=vmpywoh(Rss,Rtt)[:<<1][:rnd][:sat]
//Rdd[+]=vmpyweh(Rss,Rtt)[:<<1][:rnd][:sat]
def HEXAGON_M2_mmpyl_rs1:
  di_MInst_didi_s1_rnd_sat        <"vmpyweh",  int_hexagon_M2_mmpyl_rs1>;
def HEXAGON_M2_mmpyl_s1:
  di_MInst_didi_s1_sat            <"vmpyweh",  int_hexagon_M2_mmpyl_s1>;
def HEXAGON_M2_mmpyl_rs0:
  di_MInst_didi_rnd_sat           <"vmpyweh",  int_hexagon_M2_mmpyl_rs0>;
def HEXAGON_M2_mmpyl_s0:
  di_MInst_didi_sat               <"vmpyweh",  int_hexagon_M2_mmpyl_s0>;
def HEXAGON_M2_mmpyh_rs1:
  di_MInst_didi_s1_rnd_sat        <"vmpywoh",  int_hexagon_M2_mmpyh_rs1>;
def HEXAGON_M2_mmpyh_s1:
  di_MInst_didi_s1_sat            <"vmpywoh",  int_hexagon_M2_mmpyh_s1>;
def HEXAGON_M2_mmpyh_rs0:
  di_MInst_didi_rnd_sat           <"vmpywoh",  int_hexagon_M2_mmpyh_rs0>;
def HEXAGON_M2_mmpyh_s0:
  di_MInst_didi_sat               <"vmpywoh",  int_hexagon_M2_mmpyh_s0>;
def HEXAGON_M2_mmacls_rs1:
  di_MInst_dididi_acc_s1_rnd_sat  <"vmpyweh",  int_hexagon_M2_mmacls_rs1>;
def HEXAGON_M2_mmacls_s1:
  di_MInst_dididi_acc_s1_sat      <"vmpyweh",  int_hexagon_M2_mmacls_s1>;
def HEXAGON_M2_mmacls_rs0:
  di_MInst_dididi_acc_rnd_sat     <"vmpyweh",  int_hexagon_M2_mmacls_rs0>;
def HEXAGON_M2_mmacls_s0:
  di_MInst_dididi_acc_sat         <"vmpyweh",  int_hexagon_M2_mmacls_s0>;
def HEXAGON_M2_mmachs_rs1:
  di_MInst_dididi_acc_s1_rnd_sat  <"vmpywoh",  int_hexagon_M2_mmachs_rs1>;
def HEXAGON_M2_mmachs_s1:
  di_MInst_dididi_acc_s1_sat      <"vmpywoh",  int_hexagon_M2_mmachs_s1>;
def HEXAGON_M2_mmachs_rs0:
  di_MInst_dididi_acc_rnd_sat     <"vmpywoh",  int_hexagon_M2_mmachs_rs0>;
def HEXAGON_M2_mmachs_s0:
  di_MInst_dididi_acc_sat         <"vmpywoh",  int_hexagon_M2_mmachs_s0>;

// MTYPE / MPYH / Multiply word by unsigned half (32x16).
//Rdd[+]=vmpywouh(Rss,Rtt)[:<<1][:rnd][:sat]
//Rdd[+]=vmpyweuh(Rss,Rtt)[:<<1][:rnd][:sat]
def HEXAGON_M2_mmpyul_rs1:
  di_MInst_didi_s1_rnd_sat        <"vmpyweuh", int_hexagon_M2_mmpyul_rs1>;
def HEXAGON_M2_mmpyul_s1:
  di_MInst_didi_s1_sat            <"vmpyweuh", int_hexagon_M2_mmpyul_s1>;
def HEXAGON_M2_mmpyul_rs0:
  di_MInst_didi_rnd_sat           <"vmpyweuh", int_hexagon_M2_mmpyul_rs0>;
def HEXAGON_M2_mmpyul_s0:
  di_MInst_didi_sat               <"vmpyweuh", int_hexagon_M2_mmpyul_s0>;
def HEXAGON_M2_mmpyuh_rs1:
  di_MInst_didi_s1_rnd_sat        <"vmpywouh", int_hexagon_M2_mmpyuh_rs1>;
def HEXAGON_M2_mmpyuh_s1:
  di_MInst_didi_s1_sat            <"vmpywouh", int_hexagon_M2_mmpyuh_s1>;
def HEXAGON_M2_mmpyuh_rs0:
  di_MInst_didi_rnd_sat           <"vmpywouh", int_hexagon_M2_mmpyuh_rs0>;
def HEXAGON_M2_mmpyuh_s0:
  di_MInst_didi_sat               <"vmpywouh", int_hexagon_M2_mmpyuh_s0>;
def HEXAGON_M2_mmaculs_rs1:
  di_MInst_dididi_acc_s1_rnd_sat  <"vmpyweuh", int_hexagon_M2_mmaculs_rs1>;
def HEXAGON_M2_mmaculs_s1:
  di_MInst_dididi_acc_s1_sat      <"vmpyweuh", int_hexagon_M2_mmaculs_s1>;
def HEXAGON_M2_mmaculs_rs0:
  di_MInst_dididi_acc_rnd_sat     <"vmpyweuh", int_hexagon_M2_mmaculs_rs0>;
def HEXAGON_M2_mmaculs_s0:
  di_MInst_dididi_acc_sat         <"vmpyweuh", int_hexagon_M2_mmaculs_s0>;
def HEXAGON_M2_mmacuhs_rs1:
  di_MInst_dididi_acc_s1_rnd_sat  <"vmpywouh", int_hexagon_M2_mmacuhs_rs1>;
def HEXAGON_M2_mmacuhs_s1:
  di_MInst_dididi_acc_s1_sat      <"vmpywouh", int_hexagon_M2_mmacuhs_s1>;
def HEXAGON_M2_mmacuhs_rs0:
  di_MInst_dididi_acc_rnd_sat     <"vmpywouh", int_hexagon_M2_mmacuhs_rs0>;
def HEXAGON_M2_mmacuhs_s0:
  di_MInst_dididi_acc_sat         <"vmpywouh", int_hexagon_M2_mmacuhs_s0>;

/********************************************************************
*            MTYPE/VB                                               *
*********************************************************************/

// MTYPE / VB / Vector reduce add unsigned bytes.
def HEXAGON_A2_vraddub:
  di_MInst_didi                   <"vraddub", int_hexagon_A2_vraddub>;
def HEXAGON_A2_vraddub_acc:
  di_MInst_dididi_acc             <"vraddub", int_hexagon_A2_vraddub_acc>;

// MTYPE / VB / Vector sum of absolute differences unsigned bytes.
def HEXAGON_A2_vrsadub:
  di_MInst_didi                   <"vrsadub", int_hexagon_A2_vrsadub>;
def HEXAGON_A2_vrsadub_acc:
  di_MInst_dididi_acc             <"vrsadub", int_hexagon_A2_vrsadub_acc>;

/********************************************************************
*            MTYPE/VH                                               *
*********************************************************************/

// MTYPE / VH / Vector dual multiply.
def HEXAGON_M2_vdmpys_s1:
  di_MInst_didi_s1_sat            <"vdmpy",   int_hexagon_M2_vdmpys_s1>;
def HEXAGON_M2_vdmpys_s0:
  di_MInst_didi_sat               <"vdmpy",   int_hexagon_M2_vdmpys_s0>;
def HEXAGON_M2_vdmacs_s1:
  di_MInst_dididi_acc_s1_sat      <"vdmpy",   int_hexagon_M2_vdmacs_s1>;
def HEXAGON_M2_vdmacs_s0:
  di_MInst_dididi_acc_sat         <"vdmpy",   int_hexagon_M2_vdmacs_s0>;

// MTYPE / VH / Vector dual multiply with round and pack.
def HEXAGON_M2_vdmpyrs_s0:
  si_MInst_didi_rnd_sat           <"vdmpy",   int_hexagon_M2_vdmpyrs_s0>;
def HEXAGON_M2_vdmpyrs_s1:
  si_MInst_didi_s1_rnd_sat        <"vdmpy",   int_hexagon_M2_vdmpyrs_s1>;

// MTYPE / VH / Vector multiply even halfwords.
def HEXAGON_M2_vmpy2es_s1:
  di_MInst_didi_s1_sat            <"vmpyeh",  int_hexagon_M2_vmpy2es_s1>;
def HEXAGON_M2_vmpy2es_s0:
  di_MInst_didi_sat               <"vmpyeh",  int_hexagon_M2_vmpy2es_s0>;
def HEXAGON_M2_vmac2es:
  di_MInst_dididi_acc             <"vmpyeh",  int_hexagon_M2_vmac2es>;
def HEXAGON_M2_vmac2es_s1:
  di_MInst_dididi_acc_s1_sat      <"vmpyeh",  int_hexagon_M2_vmac2es_s1>;
def HEXAGON_M2_vmac2es_s0:
  di_MInst_dididi_acc_sat         <"vmpyeh",  int_hexagon_M2_vmac2es_s0>;

// MTYPE / VH / Vector multiply halfwords.
def HEXAGON_M2_vmpy2s_s0:
  di_MInst_sisi_sat               <"vmpyh",   int_hexagon_M2_vmpy2s_s0>;
def HEXAGON_M2_vmpy2s_s1:
  di_MInst_sisi_s1_sat            <"vmpyh",   int_hexagon_M2_vmpy2s_s1>;
def HEXAGON_M2_vmac2:
  di_MInst_disisi_acc             <"vmpyh",   int_hexagon_M2_vmac2>;
def HEXAGON_M2_vmac2s_s0:
  di_MInst_disisi_acc_sat         <"vmpyh",   int_hexagon_M2_vmac2s_s0>;
def HEXAGON_M2_vmac2s_s1:
  di_MInst_disisi_acc_s1_sat      <"vmpyh",   int_hexagon_M2_vmac2s_s1>;

// MTYPE / VH / Vector multiply halfwords with round and pack.
def HEXAGON_M2_vmpy2s_s0pack:
  si_MInst_sisi_rnd_sat           <"vmpyh",   int_hexagon_M2_vmpy2s_s0pack>;
def HEXAGON_M2_vmpy2s_s1pack:
  si_MInst_sisi_s1_rnd_sat        <"vmpyh",   int_hexagon_M2_vmpy2s_s1pack>;

// MTYPE / VH / Vector reduce multiply halfwords.
// Rxx32+=vrmpyh(Rss32,Rtt32)
def HEXAGON_M2_vrmpy_s0:
  di_MInst_didi                   <"vrmpyh",  int_hexagon_M2_vrmpy_s0>;
def HEXAGON_M2_vrmac_s0:
  di_MInst_dididi_acc             <"vrmpyh",  int_hexagon_M2_vrmac_s0>;

/********************************************************************
*            STYPE/COMPLEX                                          *
*********************************************************************/

// STYPE / COMPLEX / Vector Complex conjugate.
def HEXAGON_A2_vconj:
  di_SInst_di_sat                 <"vconj",   int_hexagon_A2_vconj>;

// STYPE / COMPLEX / Vector Complex rotate.
def HEXAGON_S2_vcrotate:
  di_SInst_disi                   <"vcrotate",int_hexagon_S2_vcrotate>;


/********************************************************************
*            STYPE/PERM                                             *
*********************************************************************/

// STYPE / PERM / Vector align.
// Need custom lowering
def HEXAGON_S2_valignib:
  di_SInst_didiu3                 <"valignb", int_hexagon_S2_valignib>;
def HEXAGON_S2_valignrb:
  di_SInst_didiqi                 <"valignb", int_hexagon_S2_valignrb>;

// STYPE / PERM / Vector round and pack.
def HEXAGON_S2_vrndpackwh:
  si_SInst_di                     <"vrndwh",  int_hexagon_S2_vrndpackwh>;
def HEXAGON_S2_vrndpackwhs:
  si_SInst_di_sat                 <"vrndwh",  int_hexagon_S2_vrndpackwhs>;

// STYPE / PERM / Vector saturate and pack.
def HEXAGON_S2_svsathb:
  si_SInst_si                     <"vsathb",  int_hexagon_S2_svsathb>;
def HEXAGON_S2_vsathb:
  si_SInst_di                     <"vsathb",  int_hexagon_S2_vsathb>;
def HEXAGON_S2_svsathub:
  si_SInst_si                     <"vsathub", int_hexagon_S2_svsathub>;
def HEXAGON_S2_vsathub:
  si_SInst_di                     <"vsathub", int_hexagon_S2_vsathub>;
def HEXAGON_S2_vsatwh:
  si_SInst_di                     <"vsatwh",  int_hexagon_S2_vsatwh>;
def HEXAGON_S2_vsatwuh:
  si_SInst_di                     <"vsatwuh", int_hexagon_S2_vsatwuh>;

// STYPE / PERM / Vector saturate without pack.
def HEXAGON_S2_vsathb_nopack:
  di_SInst_di                     <"vsathb",  int_hexagon_S2_vsathb_nopack>;
def HEXAGON_S2_vsathub_nopack:
  di_SInst_di                     <"vsathub", int_hexagon_S2_vsathub_nopack>;
def HEXAGON_S2_vsatwh_nopack:
  di_SInst_di                     <"vsatwh",  int_hexagon_S2_vsatwh_nopack>;
def HEXAGON_S2_vsatwuh_nopack:
  di_SInst_di                     <"vsatwuh", int_hexagon_S2_vsatwuh_nopack>;

// STYPE / PERM / Vector shuffle.
def HEXAGON_S2_shuffeb:
  di_SInst_didi                   <"shuffeb", int_hexagon_S2_shuffeb>;
def HEXAGON_S2_shuffeh:
  di_SInst_didi                   <"shuffeh", int_hexagon_S2_shuffeh>;
def HEXAGON_S2_shuffob:
  di_SInst_didi                   <"shuffob", int_hexagon_S2_shuffob>;
def HEXAGON_S2_shuffoh:
  di_SInst_didi                   <"shuffoh", int_hexagon_S2_shuffoh>;

// STYPE / PERM / Vector splat bytes.
def HEXAGON_S2_vsplatrb:
  si_SInst_si                     <"vsplatb", int_hexagon_S2_vsplatrb>;

// STYPE / PERM / Vector splat halfwords.
def HEXAGON_S2_vsplatrh:
  di_SInst_si                     <"vsplath", int_hexagon_S2_vsplatrh>;

// STYPE / PERM / Vector splice.
def Hexagon_S2_vsplicerb:
  di_SInst_didiqi                 <"vspliceb",int_hexagon_S2_vsplicerb>;
def Hexagon_S2_vspliceib:
  di_SInst_didiu3                 <"vspliceb",int_hexagon_S2_vspliceib>;

// STYPE / PERM / Sign extend.
def HEXAGON_S2_vsxtbh:
  di_SInst_si                     <"vsxtbh",  int_hexagon_S2_vsxtbh>;
def HEXAGON_S2_vsxthw:
  di_SInst_si                     <"vsxthw",  int_hexagon_S2_vsxthw>;

// STYPE / PERM / Truncate.
def HEXAGON_S2_vtrunehb:
  si_SInst_di                     <"vtrunehb",int_hexagon_S2_vtrunehb>;
def HEXAGON_S2_vtrunohb:
  si_SInst_di                     <"vtrunohb",int_hexagon_S2_vtrunohb>;
def HEXAGON_S2_vtrunewh:
  di_SInst_didi                   <"vtrunewh",int_hexagon_S2_vtrunewh>;
def HEXAGON_S2_vtrunowh:
  di_SInst_didi                   <"vtrunowh",int_hexagon_S2_vtrunowh>;

// STYPE / PERM / Zero extend.
def HEXAGON_S2_vzxtbh:
  di_SInst_si                     <"vzxtbh",  int_hexagon_S2_vzxtbh>;
def HEXAGON_S2_vzxthw:
  di_SInst_si                     <"vzxthw",  int_hexagon_S2_vzxthw>;


/********************************************************************
*            STYPE/PRED                                             *
*********************************************************************/

// STYPE / PRED / Mask generate from predicate.
def HEXAGON_C2_mask:
  di_SInst_qi                     <"mask",   int_hexagon_C2_mask>;

// STYPE / PRED / Predicate transfer.
def HEXAGON_C2_tfrpr:
  si_SInst_qi                     <"",       int_hexagon_C2_tfrpr>;
def HEXAGON_C2_tfrrp:
  qi_SInst_si                     <"",       int_hexagon_C2_tfrrp>;

// STYPE / PRED / Viterbi pack even and odd predicate bits.
def HEXAGON_C2_vitpack:
  si_SInst_qiqi                   <"vitpack",int_hexagon_C2_vitpack>;


/********************************************************************
*            STYPE/VH                                               *
*********************************************************************/

// STYPE / VH / Vector absolute value halfwords.
// Rdd64=vabsh(Rss64)
def HEXAGON_A2_vabsh:
  di_SInst_di                     <"vabsh",   int_hexagon_A2_vabsh>;
def HEXAGON_A2_vabshsat:
  di_SInst_di_sat                 <"vabsh",   int_hexagon_A2_vabshsat>;

// STYPE / VH / Vector shift halfwords by immediate.
// Rdd64=v[asl/asr/lsr]h(Rss64,Rt32)
def HEXAGON_S2_asl_i_vh:
  di_SInst_disi                   <"vaslh",   int_hexagon_S2_asl_i_vh>;
def HEXAGON_S2_asr_i_vh:
  di_SInst_disi                   <"vasrh",   int_hexagon_S2_asr_i_vh>;
def HEXAGON_S2_lsr_i_vh:
  di_SInst_disi                   <"vlsrh",   int_hexagon_S2_lsr_i_vh>;

// STYPE / VH / Vector shift halfwords by register.
// Rdd64=v[asl/asr/lsl/lsr]w(Rss64,Rt32)
def HEXAGON_S2_asl_r_vh:
  di_SInst_disi                   <"vaslh",   int_hexagon_S2_asl_r_vh>;
def HEXAGON_S2_asr_r_vh:
  di_SInst_disi                   <"vasrh",   int_hexagon_S2_asr_r_vh>;
def HEXAGON_S2_lsl_r_vh:
  di_SInst_disi                   <"vlslh",   int_hexagon_S2_lsl_r_vh>;
def HEXAGON_S2_lsr_r_vh:
  di_SInst_disi                   <"vlsrh",   int_hexagon_S2_lsr_r_vh>;


/********************************************************************
*            STYPE/VW                                               *
*********************************************************************/

// STYPE / VW / Vector absolute value words.
def HEXAGON_A2_vabsw:
  di_SInst_di                     <"vabsw",   int_hexagon_A2_vabsw>;
def HEXAGON_A2_vabswsat:
  di_SInst_di_sat                 <"vabsw",   int_hexagon_A2_vabswsat>;

// STYPE / VW / Vector shift words by immediate.
// Rdd64=v[asl/vsl]w(Rss64,Rt32)
def HEXAGON_S2_asl_i_vw:
  di_SInst_disi                   <"vaslw",   int_hexagon_S2_asl_i_vw>;
def HEXAGON_S2_asr_i_vw:
  di_SInst_disi                   <"vasrw",   int_hexagon_S2_asr_i_vw>;
def HEXAGON_S2_lsr_i_vw:
  di_SInst_disi                   <"vlsrw",   int_hexagon_S2_lsr_i_vw>;

// STYPE / VW / Vector shift words by register.
// Rdd64=v[asl/vsl]w(Rss64,Rt32)
def HEXAGON_S2_asl_r_vw:
  di_SInst_disi                   <"vaslw",   int_hexagon_S2_asl_r_vw>;
def HEXAGON_S2_asr_r_vw:
  di_SInst_disi                   <"vasrw",   int_hexagon_S2_asr_r_vw>;
def HEXAGON_S2_lsl_r_vw:
  di_SInst_disi                   <"vlslw",   int_hexagon_S2_lsl_r_vw>;
def HEXAGON_S2_lsr_r_vw:
  di_SInst_disi                   <"vlsrw",   int_hexagon_S2_lsr_r_vw>;

// STYPE / VW / Vector shift words with truncate and pack.
def HEXAGON_S2_asr_r_svw_trun:
  si_SInst_disi                   <"vasrw",   int_hexagon_S2_asr_r_svw_trun>;
def HEXAGON_S2_asr_i_svw_trun:
  si_SInst_diu5                   <"vasrw",   int_hexagon_S2_asr_i_svw_trun>;

// LD / Circular loads.
def HEXAGON_circ_ldd:
  di_LDInstPI_diu4                <"circ_ldd", int_hexagon_circ_ldd>;

include "HexagonIntrinsicsV3.td"
include "HexagonIntrinsicsV4.td"
include "HexagonIntrinsicsV5.td"
