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

def OP_NONE  : Op;
def OP_ADD   : Op;
def OP_SUB   : Op;
def OP_MUL   : Op;
def OP_MLA   : Op;
def OP_MLS   : Op;
def OP_MUL_N : Op;
def OP_MLA_N : Op;
def OP_MLS_N : Op;
def OP_EQ    : Op;
def OP_GE    : Op;
def OP_LE    : Op;
def OP_GT    : Op;
def OP_LT    : Op;
def OP_NEG   : Op;
def OP_NOT   : Op;
def OP_AND   : Op;
def OP_OR    : Op;
def OP_XOR   : Op;
def OP_ANDN  : Op;
def OP_ORN   : Op;
def OP_CAST  : Op;
def OP_HI    : Op;
def OP_LO    : Op;
def OP_CONC  : Op;
def OP_DUP   : Op;
def OP_SEL   : Op;
def OP_REV64 : Op;
def OP_REV32 : Op;
def OP_REV16 : Op;

class Inst <string p, string t, Op o> {
  string Prototype = p;
  string Types = t;
  Op Operand = o;
  bit isShift = 0;
}

// Used to generate Builtins.def:
// SInst: Instruction with signed/unsigned suffix (e.g., "s8", "u8", "p8")
// IInst: Instruction with generic integer suffix (e.g., "i8")
// WInst: Instruction with only bit size suffix (e.g., "8")
class SInst<string p, string t> : Inst<p, t, OP_NONE> {}
class IInst<string p, string t> : Inst<p, t, OP_NONE> {}
class WInst<string p, string t> : Inst<p, t, OP_NONE> {}

// prototype: return (arg, arg, ...)
// v: void
// t: best-fit integer (int/poly args)
// x: signed integer   (int/float args)
// u: unsigned integer (int/float args)
// f: float (int args)
// d: default
// g: default, ignore 'Q' size modifier.
// w: double width elements, same num elts
// n: double width elements, half num elts
// h: half width elements, double num elts
// e: half width elements, double num elts, unsigned
// i: constant int
// l: constant uint64
// s: scalar of element type
// a: scalar of element type (splat to vector type)
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
def VADDL   : SInst<"wdd", "csiUcUsUi">;
def VADDW   : SInst<"wwd", "csiUcUsUi">;
def VHADD   : SInst<"ddd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VRHADD  : SInst<"ddd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VQADD   : SInst<"ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VADDHN  : IInst<"dww", "csiUcUsUi">;
def VRADDHN : IInst<"dww", "csiUcUsUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.2 Multiplication
def VMUL     : Inst<"ddd",  "csifUcUsUiPcQcQsQiQfQUcQUsQUiQPc", OP_MUL>;
def VMLA     : Inst<"dddd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_MLA>;
def VMLAL    : SInst<"wwdd", "csiUcUsUi">;
def VMLS     : Inst<"dddd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_MLS>;
def VMLSL    : SInst<"wwdd", "csiUcUsUi">;
def VQDMULH  : SInst<"ddd",  "siQsQi">;
def VQRDMULH : SInst<"ddd",  "siQsQi">;
def VQDMLAL  : SInst<"wwdd", "si">;
def VQDMLSL  : SInst<"wwdd", "si">;
def VMULL    : SInst<"wdd",  "csiUcUsUiPc">;
def VQDMULL  : SInst<"wdd",  "si">;

////////////////////////////////////////////////////////////////////////////////
// E.3.3 Subtraction
def VSUB    : Inst<"ddd", "csilfUcUsUiUlQcQsQiQlQfQUcQUsQUiQUl", OP_SUB>;
def VSUBL   : SInst<"wdd", "csiUcUsUi">;
def VSUBW   : SInst<"wwd", "csiUcUsUi">;
def VQSUB   : SInst<"ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VHSUB   : SInst<"ddd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VSUBHN  : IInst<"dww", "csiUcUsUi">;
def VRSUBHN : IInst<"dww", "csiUcUsUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.4 Comparison
def VCEQ  : Inst<"udd", "csifUcUsUiPcQcQsQiQfQUcQUsQUiQPc", OP_EQ>;
def VCGE  : Inst<"udd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_GE>;
def VCLE  : Inst<"udd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_LE>;
def VCGT  : Inst<"udd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_GT>;
def VCLT  : Inst<"udd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_LT>;
def VCAGE : IInst<"udd", "fQf">;
def VCALE : IInst<"udd", "fQf">;
def VCAGT : IInst<"udd", "fQf">;
def VCALT : IInst<"udd", "fQf">;
def VTST  : WInst<"udd", "csiUcUsUiPcQcQsQiQUcQUsQUiQPc">;

////////////////////////////////////////////////////////////////////////////////
// E.3.5 Absolute Difference
def VABD  : SInst<"ddd",  "csiUcUsUifQcQsQiQUcQUsQUiQf">;
def VABDL : SInst<"wdd",  "csiUcUsUi">;
def VABA  : SInst<"dddd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VABAL : SInst<"wwdd", "csiUcUsUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.6 Max/Min
def VMAX : SInst<"ddd", "csiUcUsUifQcQsQiQUcQUsQUiQf">;
def VMIN : SInst<"ddd", "csiUcUsUifQcQsQiQUcQUsQUiQf">;

////////////////////////////////////////////////////////////////////////////////
// E.3.7 Pairdise Addition
def VPADD  : IInst<"ddd", "csiUcUsUif">;
def VPADDL : SInst<"nd",  "csiUcUsUiQcQsQiQUcQUsQUi">;
def VPADAL : SInst<"nnd", "csiUcUsUiQcQsQiQUcQUsQUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.8-9 Folding Max/Min
def VPMAX : SInst<"ddd", "csiUcUsUif">;
def VPMIN : SInst<"ddd", "csiUcUsUif">;

////////////////////////////////////////////////////////////////////////////////
// E.3.10 Reciprocal/Sqrt
def VRECPS  : IInst<"ddd", "fQf">;
def VRSQRTS : IInst<"ddd", "fQf">;

////////////////////////////////////////////////////////////////////////////////
// E.3.11 Shifts by signed variable
def VSHL   : SInst<"ddx", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VQSHL  : SInst<"ddx", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VRSHL  : SInst<"ddx", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VQRSHL : SInst<"ddx", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.12 Shifts by constant
let isShift = 1 in {
def VSHR_N     : SInst<"ddi",  "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VSHL_N     : IInst<"ddi",  "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VRSHR_N    : SInst<"ddi",  "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VSRA_N     : SInst<"dddi", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VRSRA_N    : SInst<"dddi", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VQSHL_N    : SInst<"ddi",  "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VQSHLU_N   : SInst<"udi",  "csilQcQsQiQl">;
def VSHRN_N    : IInst<"hki",  "silUsUiUl">;
def VQSHRUN_N  : SInst<"eki",  "sil">;
def VQRSHRUN_N : SInst<"eki",  "sil">;
def VQSHRN_N   : SInst<"hki",  "silUsUiUl">;
def VRSHRN_N   : IInst<"hki",  "silUsUiUl">;
def VQRSHRN_N  : SInst<"hki",  "silUsUiUl">;
def VSHLL_N    : SInst<"wdi",  "csiUcUsUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.13 Shifts with insert
def VSRI_N : WInst<"dddi", "csilUcUsUiUlPcPsQcQsQiQlQUcQUsQUiQUlQPcQPs">;
def VSLI_N : WInst<"dddi", "csilUcUsUiUlPcPsQcQsQiQlQUcQUsQUiQUlQPcQPs">;
}

////////////////////////////////////////////////////////////////////////////////
// E.3.14 Loads and stores of a single vector
def VLD1      : WInst<"dc",   "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD1_LANE : WInst<"dcdi", "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD1_DUP  : WInst<"dc",   "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST1      : WInst<"vpd",  "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST1_LANE : WInst<"vpdi", "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.15 Loads and stores of an N-element structure
def VLD2      : WInst<"2c",   "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD3      : WInst<"3c",   "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD4      : WInst<"4c",   "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD2_DUP  : WInst<"2c",   "UcUsUiUlcsilhfPcPs">;
def VLD3_DUP  : WInst<"3c",   "UcUsUiUlcsilhfPcPs">;
def VLD4_DUP  : WInst<"4c",   "UcUsUiUlcsilhfPcPs">;
def VLD2_LANE : WInst<"2c2i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VLD3_LANE : WInst<"3c3i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VLD4_LANE : WInst<"4c4i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VST2      : WInst<"vp2",  "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST3      : WInst<"vp3",  "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST4      : WInst<"vp4",  "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST2_LANE : WInst<"vp2i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VST3_LANE : WInst<"vp3i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VST4_LANE : WInst<"vp4i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.16 Extract lanes from a vector
def VGET_LANE : IInst<"sdi", "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.17 Set lanes within a vector
def VSET_LANE : IInst<"dsdi", "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.18 Initialize a vector from bit pattern
def VCREATE: Inst<"dl", "csihfUcUsUiUlPcPsl", OP_CAST>;

////////////////////////////////////////////////////////////////////////////////
// E.3.19 Set all lanes to same value
def VDUP_N : Inst<"ds", "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl", OP_DUP>;
def VMOV_N : Inst<"ds", "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl", OP_DUP>;
def VDUP_LANE : WInst<"dgi", "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.20 Combining vectors
def VCOMBINE : Inst<"kdd", "csilhfUcUsUiUlPcPs", OP_CONC>;

////////////////////////////////////////////////////////////////////////////////
// E.3.21 Splitting vectors
def VGET_HIGH : Inst<"dk", "csilhfUcUsUiUlPcPs", OP_HI>;
def VGET_LOW  : Inst<"dk", "csilhfUcUsUiUlPcPs", OP_LO>;

////////////////////////////////////////////////////////////////////////////////
// E.3.22 Converting vectors
def VCVT_S32     : SInst<"xd",  "fQf">;
def VCVT_U32     : SInst<"ud",  "fQf">;
def VCVT_F16     : SInst<"hk",  "f">;
def VCVT_N_S32   : SInst<"xdi", "fQf">;
def VCVT_N_U32   : SInst<"udi", "fQf">;
def VCVT_F32     : SInst<"fd",  "iUiQiQUi">;
def VCVT_F32_F16 : SInst<"kh",  "f">;
def VCVT_N_F32   : SInst<"fdi", "iUiQiQUi">;
def VMOVN        : IInst<"hk",  "silUsUiUl">;
def VMOVL        : SInst<"wd",  "csiUcUsUi">;
def VQMOVN       : SInst<"hk",  "silUsUiUl">;
def VQMOVUN      : SInst<"ek",  "sil">;

////////////////////////////////////////////////////////////////////////////////
// E.3.23-24 Table lookup, Extended table lookup
def VTBL1 : WInst<"ddt",  "UccPc">;
def VTBL2 : WInst<"d2t",  "UccPc">;
def VTBL3 : WInst<"d3t",  "UccPc">;
def VTBL4 : WInst<"d4t",  "UccPc">;
def VTBX1 : WInst<"dddt", "UccPc">;
def VTBX2 : WInst<"dd2t", "UccPc">;
def VTBX3 : WInst<"dd3t", "UccPc">;
def VTBX4 : WInst<"dd4t", "UccPc">;

////////////////////////////////////////////////////////////////////////////////
// E.3.25 Operations with a scalar value
def VMLA_LANE     : IInst<"ddddi", "siUsUifQsQiQUsQUiQf">;
def VMLAL_LANE    : SInst<"wwddi", "siUsUi">;
def VQDMLAL_LANE  : SInst<"wwddi", "si">; 
def VMLS_LANE     : IInst<"ddddi", "siUsUifQsQiQUsQUiQf">;
def VMLSL_LANE    : SInst<"wwddi", "siUsUi">;
def VQDMLSL_LANE  : SInst<"wwddi", "si">;
def VMUL_N        : Inst<"dds",    "sifUsUiQsQiQfQUsQUi", OP_MUL_N>;
def VMULL_N       : SInst<"wda",   "siUsUi">;
def VMULL_LANE    : SInst<"wddi",  "siUsUi">;
def VQDMULL_N     : SInst<"wda",   "si">;
def VQDMULL_LANE  : SInst<"wddi",  "si">;
def VQDMULH_N     : SInst<"dda",   "siQsQi">;
def VQDMULH_LANE  : SInst<"dddi",  "siQsQi">;
def VQRDMULH_N    : SInst<"dda",   "siQsQi">;
def VQRDMULH_LANE : SInst<"dddi",  "siQsQi">;
def VMLA_N        : Inst<"ddda",   "siUsUifQsQiQUsQUiQf", OP_MLA_N>;
def VMLAL_N       : SInst<"wwda",  "siUsUi">;
def VQDMLAL_N     : SInst<"wwda",  "si">;
def VMLS_N        : Inst<"ddds",   "siUsUifQsQiQUsQUiQf", OP_MLS_N>;
def VMLSL_N       : SInst<"wwda",  "siUsUi">;
def VQDMLSL_N     : SInst<"wwda",  "si">;

////////////////////////////////////////////////////////////////////////////////
// E.3.26 Vector Extract
def VEXT : WInst<"dddi", "cUcPcsUsPsiUilUlQcQUcQPcQsQUsQPsQiQUiQlQUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.27 Reverse vector elements (sdap endianness)
def VREV64 : Inst<"dd", "csiUcUsUiPcPsfQcQsQiQUcQUsQUiQPcQPsQf", OP_REV64>;
def VREV32 : Inst<"dd", "csUcUsPcQcQsQUcQUsQPc", OP_REV32>;
def VREV16 : Inst<"dd", "cUcPcQcQUcQPc", OP_REV16>;

////////////////////////////////////////////////////////////////////////////////
// E.3.28 Other single operand arithmetic
def VABS    : SInst<"dd", "csifQcQsQiQf">;
def VQABS   : SInst<"dd", "csiQcQsQi">;
def VNEG    : Inst<"dd",  "csifQcQsQiQf", OP_NEG>;
def VQNEG   : SInst<"dd", "csiQcQsQi">;
def VCLS    : SInst<"dd", "csiQcQsQi">;
def VCLZ    : IInst<"dd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VCNT    : WInst<"dd", "UccPcQUcQcQPc">;
def VRECPE  : SInst<"dd", "fUiQfQUi">;
def VRSQRTE : SInst<"dd", "fUiQfQUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.29 Logical operations
def VMVN : Inst<"dd",    "csiUcUsUiPcQcQsQiQUcQUsQUiQPc", OP_NOT>;
def VAND : Inst<"ddd",   "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_AND>;
def VORR : Inst<"ddd",   "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_OR>;
def VEOR : Inst<"ddd",   "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_XOR>;
def VBIC : Inst<"ddd",   "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_ANDN>;
def VORN : Inst<"ddd",   "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_ORN>;
def VBSL : Inst<"dudd",  "csilUcUsUiUlfPcPsQcQsQiQlQUcQUsQUiQUlQfQPcQPs", OP_SEL>;

////////////////////////////////////////////////////////////////////////////////
// E.3.30 Transposition operations
def VTRN: WInst<"2dd", "csiUcUsUifPcPsQcQsQiQUcQUsQUiQfQPcQPs">;
def VZIP: WInst<"2dd", "csUcUsfPcPsQcQsQiQUcQUsQUiQfQPcQPs">;
def VUZP: WInst<"2dd", "csiUcUsUifPcPsQcQsQiQUcQUsQUiQfQPcQPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.31 Vector reinterpret cast operations
