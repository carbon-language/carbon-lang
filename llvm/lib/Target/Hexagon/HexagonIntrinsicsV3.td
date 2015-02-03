//=- HexagonIntrinsicsV3.td - Target Description for Hexagon -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon V3 Compiler Intrinsics in TableGen format.
//
//===----------------------------------------------------------------------===//

// Vector reduce complex multiply real or imaginary
def : T_PR_pat <M2_vrcmpys_s1,     int_hexagon_M2_vrcmpys_s1>;
def : T_PPR_pat<M2_vrcmpys_acc_s1, int_hexagon_M2_vrcmpys_acc_s1>;
def : T_PR_pat <M2_vrcmpys_s1rp,   int_hexagon_M2_vrcmpys_s1rp>;

// Vector reduce add unsigned halfwords
def : T_PP_pat<M2_vradduh, int_hexagon_M2_vradduh>;

def: T_RP_pat<A2_addsp,   int_hexagon_A2_addsp>;
def: T_PP_pat<A2_addpsat, int_hexagon_A2_addpsat>;
def: T_PP_pat<A2_minp,    int_hexagon_A2_minp>;
def: T_PP_pat<A2_minup,   int_hexagon_A2_minup>;
def: T_PP_pat<A2_maxp,    int_hexagon_A2_maxp>;
def: T_PP_pat<A2_maxup,   int_hexagon_A2_maxup>;
