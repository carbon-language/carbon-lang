//===--- arm_neon.td - ARM NEON compiler interface ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TableGen definitions from which the ARM NEON header
//  file will be generated.  See ARM document DUI0348B.
//
//===----------------------------------------------------------------------===//

class Op;

def OP_NONE : Op;
def OP_ADD  : Op;
def OP_SUB  : Op;
def OP_MUL  : Op;
def OP_MLA  : Op;
def OP_MLS  : Op;
def OP_EQ   : Op;
def OP_GE   : Op;
def OP_LE   : Op;
def OP_GT   : Op;
def OP_LT   : Op;
def OP_NEG  : Op;
def OP_NOT  : Op;
def OP_AND  : Op;
def OP_OR   : Op;
def OP_XOR  : Op;
def OP_ANDN : Op;
def OP_ORN  : Op;

class Inst <string p, string t, Op o = OP_NONE> {
  string Prototype = p;
  string Types = t;
  Op Operand = o;
}

// prototype: return (arg, arg, ...)
// v: void
// t: best-fit integer (int/poly args)
// x: unsigned integer (int/float args)
// f: float (int args)
// d: default
// w: double width elements, same num elts
// n: double width elements, half num elts
// i: constant int
// l: constant uint64
// s: scalar of element type
// k: default elt width, double num elts
// #: array of default vectors
// p: pointer type
// c: const pointer type

// sizes:
// c: char
// s: short
// i: int
// l: long
// f: float
// h: half-float

// size modifiers:
// U: unsigned
// Q: 128b
// P: polynomial

////////////////////////////////////////////////////////////////////////////////
// E.3.1 Addition
def VADD    : Inst<"ddd", "csilfUcUsUiUlQcQsQiQlQfQUcQUsQUiQUl", OP_ADD>;
def VADDL   : Inst<"wdd", "csiUcUsUi">;
def VADDw   : Inst<"wwd", "csiUcUsUi">;
def VHADD   : Inst<"ddd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VRHADD  : Inst<"ddd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VQADD   : Inst<"ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VADDHN  : Inst<"dww", "csiUcUsUi">;
def VRADDHN : Inst<"dww", "csiUcUsUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.2 Multiplication
def VMUL     : Inst<"ddd",  "csifUcUsUiPcQcQsQiQfQUcQUsQUiQPc", OP_MUL>;
def VMLA     : Inst<"dddd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_MLA>;
def VMLAL    : Inst<"wwdd", "csiUcUsUi">;
def VMLS     : Inst<"dddd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_MLS>;
def VMLSL    : Inst<"wwdd", "csiUcUsUi">;
def VQDMULH  : Inst<"ddd",  "siQsQi">;
def VQRDMULH : Inst<"ddd",  "siQsQi">;
def VQDMLAL  : Inst<"wwdd", "si">;
def VQDMLSL  : Inst<"wwdd", "si">;
def VMULL    : Inst<"wdd",  "csiUcUsUiPc">;
def VQDMULL  : Inst<"wdd",  "si">;

////////////////////////////////////////////////////////////////////////////////
// E.3.3 Subtraction
def VSUB    : Inst<"ddd", "csilfUcUsUiUlQcQsQiQlQfQUcQUsQUiQUl", OP_SUB>;
def VSUBL   : Inst<"wdd", "csiUcUsUi">;
def VSUBw   : Inst<"wwd", "csiUcUsUi">;
def VQSUB   : Inst<"ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VHSUB   : Inst<"ddd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VSUBHN  : Inst<"dww", "csiUcUsUi">;
def VRSUBHN : Inst<"dww", "csiUcUsUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.4 Comparison
def VCEQ  : Inst<"xdd", "csifUcUsUiPcQcQsQiQfQUcQUsQUiQPc", OP_EQ>;
def VCGE  : Inst<"xdd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_GE>;
def VCLE  : Inst<"xdd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_LE>;
def VCGT  : Inst<"xdd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_GT>;
def VCLT  : Inst<"xdd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_LT>;
def VCAGE : Inst<"xdd", "fQf">;
def VCALE : Inst<"xdd", "fQf">;
def VCAGT : Inst<"xdd", "fQf">;
def VCALT : Inst<"xdd", "fQf">;
def VTST  : Inst<"xdd", "csiUcUsUiPcQcQsQiQUcQUsQUiQPc">;

////////////////////////////////////////////////////////////////////////////////
// E.3.5 Absolute Difference
def VABD  : Inst<"ddd",  "csiUcUsUifQcQsQiQUcQUsQUiQf">;
def VABDL : Inst<"wdd",  "csiUcUsUi">;
def VABA  : Inst<"dddd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VABAL : Inst<"wwdd", "csiUcUsUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.6 Max/Min
def VMAX : Inst<"ddd", "csiUcUsUifQcQsQiQUcQUsQUiQf">;
def VMIN : Inst<"ddd", "csiUcUsUifQcQsQiQUcQUsQUiQf">;

////////////////////////////////////////////////////////////////////////////////
// E.3.7 Pairdise Addition
def VPADD  : Inst<"ddd", "csiUcUsUif">;
def VPADDL : Inst<"nd",  "csiUcUsUiQcQsQiQUcQUsQUi">;
def VPADAL : Inst<"nnd", "csiUcUsUiQcQsQiQUcQUsQUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.8-9 Folding Max/Min
def VPMAX : Inst<"ddd", "csiUcUsUif">;
def VPMIN : Inst<"ddd", "csiUcUsUif">;

////////////////////////////////////////////////////////////////////////////////
// E.3.10 Reciprocal/Sqrt
def VRECPS  : Inst<"ddd", "fQf">;
def VRSQRTS : Inst<"ddd", "fQf">;

////////////////////////////////////////////////////////////////////////////////
// E.3.11 Shifts by signed variable
def VSHL   : Inst<"ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VQSHL  : Inst<"ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VRSHL  : Inst<"ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VQRSHL : Inst<"ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.12 Shifts by constant
def VSHR_N     : Inst<"ddi",  "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VSHL_N     : Inst<"ddi",  "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VRSHR_N    : Inst<"ddi",  "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VSRA_N     : Inst<"dddi", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VRSRA_N    : Inst<"dddi", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VQSHL_N    : Inst<"ddi",  "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VQSHLU_N   : Inst<"xdi",  "csilQcQsQiQl">;
def VSHRN_N    : Inst<"dwi",  "csiUcUsUi">;
def VQSHRUN_N  : Inst<"xwi",  "csi">;
def VQRSHRUN_N : Inst<"xwi",  "csi">;
def VQSHRN_N   : Inst<"dwi",  "csiUcUsUi">;
def VRSHRN_N   : Inst<"dwi",  "csiUcUsUi">;
def VQRSHRN_N  : Inst<"dwi",  "csiUcUsUi">;
def VSHLL_N    : Inst<"wdi",  "csiUcUsUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.13 Shifts with insert
def VSRI_N : Inst<"dddi", "csilUcUsUiUlPcPsQcQsQiQlQUcQUsQUiQUlQPcQPs">;
def VSLI_N : Inst<"dddi", "csilUcUsUiUlPcPsQcQsQiQlQUcQUsQUiQUlQPcQPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.14 Loads and stores of a single vector
def VLD1      : Inst<"dc",   "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD1_LANE : Inst<"dci",  "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD1_DUP  : Inst<"dc",   "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST1      : Inst<"vpd",  "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST1_LANE : Inst<"vpdi", "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.15 Loads and stores of an N-element structure
def VLD2      : Inst<"2c",   "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD3      : Inst<"3c",   "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD4      : Inst<"4c",   "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD2_DUP  : Inst<"2c",   "UcUsUiUlcsilhfPcPs">;
def VLD3_DUP  : Inst<"3c",   "UcUsUiUlcsilhfPcPs">;
def VLD4_DUP  : Inst<"4c",   "UcUsUiUlcsilhfPcPs">;
def VLD2_LANE : Inst<"2ci",  "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VLD3_LANE : Inst<"3ci",  "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VLD4_LANE : Inst<"4ci",  "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VST2      : Inst<"vp2",  "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST3      : Inst<"vp3",  "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST4      : Inst<"vp4",  "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST2_LANE : Inst<"vp2i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VST3_LANE : Inst<"vp3i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VST4_LANE : Inst<"vp4i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.16 Extract lanes from a vector
def VGET_LANE : Inst<"ddi", "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.17 Set lanes within a vector
def VSET_LANE : Inst<"dddi", "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.18 Initialize a vector from bit pattern
def VCREATE: Inst<"dl", "csihfUcUsUiUlPcPsl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.19 Set all lanes to same value
def VDUP_N : Inst<"ds", "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl">;
def VMOV_N : Inst<"ds", "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.20 Combining vectors
def VCOMBINE : Inst<"kdd", "csilhfUcUsUiUlPcPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.21 Splitting vectors
def VGET_HIGH : Inst<"dk", "csilhfUcUsUiUlPcPs">;
def VGET_LOW  : Inst<"dk", "csilhfUcUsUiUlPcPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.22 Converting vectors
def VCVT_S32   : Inst<"df",  "iQi">;
def VCVT_U32   : Inst<"df",  "UiQUi">;
def VCVT_F16   : Inst<"df",  "h">;
def VCVT_N_S32 : Inst<"dfi", "iQi">;
def VCVT_N_U32 : Inst<"dfi", "UiQUi">;
def VCVT_F32   : Inst<"fd",  "iUiQiQUih">;
def VCVT_N_F32 : Inst<"fdi", "iUiQiQUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.23-24 Table lookup, Extended table lookup
def VTBL1 : Inst<"ddt",  "UccPc">;
def VTBL2 : Inst<"d2t",  "UccPc">;
def VTBL3 : Inst<"d3t",  "UccPc">;
def VTBL4 : Inst<"d4t",  "UccPc">;
def VTBX1 : Inst<"dddt", "UccPc">;
def VTBX2 : Inst<"dd2t", "UccPc">;
def VTBX3 : Inst<"dd3t", "UccPc">;
def VTBX4 : Inst<"dd4t", "UccPc">;

////////////////////////////////////////////////////////////////////////////////
// E.3.25 Operations with a scalar value
def VMLA_LANE     : Inst<"ddddi", "siUsUifQsQiQUsQUiQf">;
def VMLAL_LANE    : Inst<"wwddi", "siUsUi">;
def VQDMLAL_LANE  : Inst<"wwddi", "si">; 
def VMLS_LANE     : Inst<"ddddi", "siUsUifQsQiQUsQUiQf">;
def VMLSL_LANE    : Inst<"wwddi", "siUsUi">;
def VQDMLSL_LANE  : Inst<"wwddi", "si">;
def VMUL_N        : Inst<"dds",   "sifUsUiQsQiQfQUsQUi">;
def VMULL_N       : Inst<"wds",   "siUsUi">;
def VMULL_LANE    : Inst<"wddi",  "siUsUi">;
def VQDMULL_N     : Inst<"wds",   "si">;
def VQDMULL_LANE  : Inst<"wddi",  "si">;
def VQDMULH_N     : Inst<"dds",   "siQsQi">;
def VQDMULH_LANE  : Inst<"dddi",  "siQsQi">;
def VQRDMULH_N    : Inst<"dds",   "siQsQi">;
def VQRDMULH_LANE : Inst<"dddi",  "siQsQi">;
def VMLA_N        : Inst<"ddds",  "siUsUifQsQiQUsQUiQf">;
def VQDMLAL_N     : Inst<"wwds",  "si">;
def VMLS_N        : Inst<"ddds",  "siUsUifQsQiQUsQUiQf">;
def VMLSL_N       : Inst<"wwds",  "siUsUi">;
def VQDMLSL_N     : Inst<"wwds",  "si">;

////////////////////////////////////////////////////////////////////////////////
// E.3.26 Vector Extract
def VEXT : Inst<"dddi", "cUcPcsUsPsiUilUlQcQUcQPcQsQUsQPsQiQUiQlQUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.27 Reverse vector elements (sdap endianness)
def VREV64 : Inst<"dd", "csiUcUsUiPcPsfQcQsQiQUcQUsQUiQPcQPsQf">;
def VREV32 : Inst<"dd", "csUcUsPcQcQsQUcQUsQPc">;
def VREV16 : Inst<"dd", "cUcPcQcQUcQPc">;

////////////////////////////////////////////////////////////////////////////////
// E.3.28 Other single operand arithmetic
def VABS    : Inst<"dd", "csifQcQsQiQf">;
def VQABS   : Inst<"dd", "csiQcQsQi">;
def VNEG    : Inst<"dd", "csifQcQsQiQf", OP_NEG>;
def VQNEG   : Inst<"dd", "csiQcQsQi">;
def VCLS    : Inst<"dd", "csiQcQsQi">;
def VCLZ    : Inst<"dd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VCNT    : Inst<"dd", "UccPcQUcQcQPc">;
def VRECPE  : Inst<"dd", "fUiQfQUi">;
def VRSQRTE : Inst<"dd", "fUiQfQUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.29 Logical operations
def VMVN : Inst<"dd",   "csiUcUsUiPcQcQsQiQUcQUsQUiQPc", OP_NOT>;
def VAND : Inst<"ddd",  "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_AND>;
def VORR : Inst<"ddd",  "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_OR>;
def VEOR : Inst<"ddd",  "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_XOR>;
def VBIC : Inst<"ddd",  "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_ANDN>;
def VORN : Inst<"ddd",  "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_ORN>;
def VBSL : Inst<"dxdd", "csilUcUsUiUlfPcPsQcQsQiQlQUcQUsQUiQUlQfQPcQPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.30 Transposition operations
def VTRN: Inst<"2dd", "csiUcUsUifPcPsQcQsQiQUcQUsQUiQfQPcQPs">;
def VZIP: Inst<"2dd", "csUcUsfPcPsQcQsQiQUcQUsQUiQfQPcQPs">;
def VUZP: Inst<"2dd", "csiUcUsUifPcPsQcQsQiQUcQUsQUiQfQPcQPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.31 Vector reinterpret cast operations
