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




// MTYPE / COMPLEX / Vector reduce complex multiply real or imaginary.
def Hexagon_M2_vrcmpys_s1:
  di_MInst_disi_s1_sat            <"vrcmpys",  int_hexagon_M2_vrcmpys_s1>;
def Hexagon_M2_vrcmpys_acc_s1:
  di_MInst_didisi_acc_s1_sat      <"vrcmpys",  int_hexagon_M2_vrcmpys_acc_s1>;
def Hexagon_M2_vrcmpys_s1rp:
  si_MInst_disi_s1_rnd_sat        <"vrcmpys",  int_hexagon_M2_vrcmpys_s1rp>;


/********************************************************************
*            MTYPE/VB                                               *
*********************************************************************/

// MTYPE / VB / Vector reduce add unsigned bytes.
def Hexagon_M2_vradduh:
  si_MInst_didi                   <"vradduh",  int_hexagon_M2_vradduh>;


def: T_RP_pat<A2_addsp,   int_hexagon_A2_addsp>;
def: T_PP_pat<A2_addpsat, int_hexagon_A2_addpsat>;
def: T_PP_pat<A2_minp,    int_hexagon_A2_minp>;
def: T_PP_pat<A2_minup,   int_hexagon_A2_minup>;
def: T_PP_pat<A2_maxp,    int_hexagon_A2_maxp>;
def: T_PP_pat<A2_maxup,   int_hexagon_A2_maxup>;
