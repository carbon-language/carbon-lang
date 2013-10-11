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
def OP_UNAVAILABLE : Op;
def OP_ADD   : Op;
def OP_ADDL  : Op;
def OP_ADDLHi : Op;
def OP_ADDW  : Op;
def OP_ADDWHi : Op;
def OP_SUB   : Op;
def OP_SUBL  : Op;
def OP_SUBLHi : Op;
def OP_SUBW  : Op;
def OP_SUBWHi : Op;
def OP_MUL   : Op;
def OP_MLA   : Op;
def OP_MLAL  : Op;
def OP_MULLHi : Op;
def OP_MLALHi : Op;
def OP_MLS   : Op;
def OP_MLSL  : Op;
def OP_MLSLHi : Op;
def OP_MUL_N : Op;
def OP_MLA_N : Op;
def OP_MLS_N : Op;
def OP_MLAL_N : Op;
def OP_MLSL_N : Op;
def OP_MUL_LN: Op;
def OP_MULX_LN: Op;
def OP_MULL_LN : Op;
def OP_MULLHi_LN : Op;
def OP_MLA_LN: Op;
def OP_MLS_LN: Op;
def OP_MLAL_LN : Op;
def OP_MLALHi_LN : Op;
def OP_MLSL_LN : Op;
def OP_MLSLHi_LN : Op;
def OP_QDMULL_LN : Op;
def OP_QDMULLHi_LN : Op;
def OP_QDMLAL_LN : Op;
def OP_QDMLALHi_LN : Op;
def OP_QDMLSL_LN : Op;
def OP_QDMLSLHi_LN : Op;
def OP_QDMULH_LN : Op;
def OP_QRDMULH_LN : Op;
def OP_FMS_LN : Op;
def OP_FMS_LNQ : Op;
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
def OP_DUP_LN: Op;
def OP_SEL   : Op;
def OP_REV64 : Op;
def OP_REV32 : Op;
def OP_REV16 : Op;
def OP_REINT : Op;
def OP_ADDHNHi : Op;
def OP_RADDHNHi : Op;
def OP_SUBHNHi : Op;
def OP_RSUBHNHi : Op;
def OP_ABDL  : Op;
def OP_ABDLHi : Op;
def OP_ABA   : Op;
def OP_ABAL  : Op;
def OP_ABALHi : Op;
def OP_QDMULLHi : Op;
def OP_QDMLALHi : Op;
def OP_QDMLSLHi : Op;
def OP_DIV  : Op;
def OP_LONG_HI : Op;
def OP_NARROW_HI : Op;
def OP_MOVL_HI : Op;
def OP_COPY : Op;

class Inst <string n, string p, string t, Op o> {
  string Name = n;
  string Prototype = p;
  string Types = t;
  Op Operand = o;
  bit isShift = 0;
  bit isVCVT_N = 0;
  bit isA64 = 0;

  // Certain intrinsics have different names than their representative
  // instructions. This field allows us to handle this correctly when we
  // are generating tests.
  string InstName = "";

  // Certain intrinsics even though they are not a WOpInst or LOpInst,
  // generate a WOpInst/LOpInst instruction (see below for definition
  // of a WOpInst/LOpInst). For testing purposes we need to know
  // this. Ex: vset_lane which outputs vmov instructions.
  bit isHiddenWInst = 0;
  bit isHiddenLInst = 0;
}

// The following instruction classes are implemented via builtins.
// These declarations are used to generate Builtins.def:
//
// SInst: Instruction with signed/unsigned suffix (e.g., "s8", "u8", "p8")
// IInst: Instruction with generic integer suffix (e.g., "i8")
// WInst: Instruction with only bit size suffix (e.g., "8")
class SInst<string n, string p, string t> : Inst<n, p, t, OP_NONE> {}
class IInst<string n, string p, string t> : Inst<n, p, t, OP_NONE> {}
class WInst<string n, string p, string t> : Inst<n, p, t, OP_NONE> {}

// The following instruction classes are implemented via operators
// instead of builtins. As such these declarations are only used for
// the purpose of generating tests.
//
// SOpInst:       Instruction with signed/unsigned suffix (e.g., "s8",
//                "u8", "p8").
// IOpInst:       Instruction with generic integer suffix (e.g., "i8").
// WOpInst:       Instruction with bit size only suffix (e.g., "8").
// LOpInst:       Logical instruction with no bit size suffix.
// NoTestOpInst:  Intrinsic that has no corresponding instruction.
class SOpInst<string n, string p, string t, Op o> : Inst<n, p, t, o> {}
class IOpInst<string n, string p, string t, Op o> : Inst<n, p, t, o> {}
class WOpInst<string n, string p, string t, Op o> : Inst<n, p, t, o> {}
class LOpInst<string n, string p, string t, Op o> : Inst<n, p, t, o> {}
class NoTestOpInst<string n, string p, string t, Op o> : Inst<n, p, t, o> {}

// prototype: return (arg, arg, ...)
// v: void
// t: best-fit integer (int/poly args)
// x: signed integer   (int/float args)
// u: unsigned integer (int/float args)
// f: float (int args)
// d: default
// g: default, ignore 'Q' size modifier.
// j: default, force 'Q' size modifier.
// w: double width elements, same num elts
// n: double width elements, half num elts
// h: half width elements, double num elts
// q: half width elements, quad num elts
// e: half width elements, double num elts, unsigned
// m: half width elements, same num elts
// i: constant int
// l: constant uint64
// s: scalar of element type
// r: scalar of double width element type
// a: scalar of element type (splat to vector type)
// y: scalar of float
// o: scalar of double
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
// d: double

// size modifiers:
// S: scalar, only used for function mangling.
// U: unsigned
// Q: 128b
// H: 128b without mangling 'q'
// P: polynomial

////////////////////////////////////////////////////////////////////////////////
// E.3.1 Addition
def VADD    : IOpInst<"vadd", "ddd",
                      "csilfUcUsUiUlQcQsQiQlQfQUcQUsQUiQUl", OP_ADD>;
def VADDL   : SOpInst<"vaddl", "wdd", "csiUcUsUi", OP_ADDL>;
def VADDW   : SOpInst<"vaddw", "wwd", "csiUcUsUi", OP_ADDW>;
def VHADD   : SInst<"vhadd", "ddd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VRHADD  : SInst<"vrhadd", "ddd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VQADD   : SInst<"vqadd", "ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VADDHN  : IInst<"vaddhn", "hkk", "silUsUiUl">;
def VRADDHN : IInst<"vraddhn", "hkk", "silUsUiUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.2 Multiplication
def VMUL     : IOpInst<"vmul", "ddd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_MUL>;
def VMULP    : SInst<"vmul", "ddd", "PcQPc">;
def VMLA     : IOpInst<"vmla", "dddd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_MLA>;
def VMLAL    : SOpInst<"vmlal", "wwdd", "csiUcUsUi", OP_MLAL>;
def VMLS     : IOpInst<"vmls", "dddd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_MLS>;
def VMLSL    : SOpInst<"vmlsl", "wwdd", "csiUcUsUi", OP_MLSL>;
def VQDMULH  : SInst<"vqdmulh", "ddd", "siQsQi">;
def VQRDMULH : SInst<"vqrdmulh", "ddd", "siQsQi">;
def VQDMLAL  : SInst<"vqdmlal", "wwdd", "si">;
def VQDMLSL  : SInst<"vqdmlsl", "wwdd", "si">;
def VMULL    : SInst<"vmull", "wdd", "csiUcUsUiPc">;
def VQDMULL  : SInst<"vqdmull", "wdd", "si">;

////////////////////////////////////////////////////////////////////////////////
// E.3.3 Subtraction
def VSUB    : IOpInst<"vsub", "ddd",
                      "csilfUcUsUiUlQcQsQiQlQfQUcQUsQUiQUl", OP_SUB>;
def VSUBL   : SOpInst<"vsubl", "wdd", "csiUcUsUi", OP_SUBL>;
def VSUBW   : SOpInst<"vsubw", "wwd", "csiUcUsUi", OP_SUBW>;
def VQSUB   : SInst<"vqsub", "ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VHSUB   : SInst<"vhsub", "ddd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VSUBHN  : IInst<"vsubhn", "hkk", "silUsUiUl">;
def VRSUBHN : IInst<"vrsubhn", "hkk", "silUsUiUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.4 Comparison
def VCEQ  : IOpInst<"vceq", "udd", "csifUcUsUiPcQcQsQiQfQUcQUsQUiQPc", OP_EQ>;
def VCGE  : SOpInst<"vcge", "udd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_GE>;
let InstName = "vcge" in
def VCLE  : SOpInst<"vcle", "udd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_LE>;
def VCGT  : SOpInst<"vcgt", "udd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_GT>;
let InstName = "vcgt" in
def VCLT  : SOpInst<"vclt", "udd", "csifUcUsUiQcQsQiQfQUcQUsQUi", OP_LT>;
let InstName = "vacge" in {
def VCAGE : IInst<"vcage", "udd", "fQf">;
def VCALE : IInst<"vcale", "udd", "fQf">;
}
let InstName = "vacgt" in {
def VCAGT : IInst<"vcagt", "udd", "fQf">;
def VCALT : IInst<"vcalt", "udd", "fQf">;
}
def VTST  : WInst<"vtst", "udd", "csiUcUsUiPcPsQcQsQiQUcQUsQUiQPcQPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.5 Absolute Difference
def VABD  : SInst<"vabd", "ddd",  "csiUcUsUifQcQsQiQUcQUsQUiQf">;
def VABDL : SOpInst<"vabdl", "wdd",  "csiUcUsUi", OP_ABDL>;
def VABA  : SOpInst<"vaba", "dddd", "csiUcUsUiQcQsQiQUcQUsQUi", OP_ABA>;
def VABAL : SOpInst<"vabal", "wwdd", "csiUcUsUi", OP_ABAL>;

////////////////////////////////////////////////////////////////////////////////
// E.3.6 Max/Min
def VMAX : SInst<"vmax", "ddd", "csiUcUsUifQcQsQiQUcQUsQUiQf">;
def VMIN : SInst<"vmin", "ddd", "csiUcUsUifQcQsQiQUcQUsQUiQf">;

////////////////////////////////////////////////////////////////////////////////
// E.3.7 Pairwise Addition
def VPADD  : IInst<"vpadd", "ddd", "csiUcUsUif">;
def VPADDL : SInst<"vpaddl", "nd",  "csiUcUsUiQcQsQiQUcQUsQUi">;
def VPADAL : SInst<"vpadal", "nnd", "csiUcUsUiQcQsQiQUcQUsQUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.8-9 Folding Max/Min
def VPMAX : SInst<"vpmax", "ddd", "csiUcUsUif">;
def VPMIN : SInst<"vpmin", "ddd", "csiUcUsUif">;

////////////////////////////////////////////////////////////////////////////////
// E.3.10 Reciprocal/Sqrt
def VRECPS  : IInst<"vrecps", "ddd", "fQf">;
def VRSQRTS : IInst<"vrsqrts", "ddd", "fQf">;

////////////////////////////////////////////////////////////////////////////////
// E.3.11 Shifts by signed variable
def VSHL   : SInst<"vshl", "ddx", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VQSHL  : SInst<"vqshl", "ddx", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VRSHL  : SInst<"vrshl", "ddx", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VQRSHL : SInst<"vqrshl", "ddx", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.12 Shifts by constant
let isShift = 1 in {
def VSHR_N     : SInst<"vshr_n", "ddi", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VSHL_N     : IInst<"vshl_n", "ddi", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VRSHR_N    : SInst<"vrshr_n", "ddi", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VSRA_N     : SInst<"vsra_n", "dddi", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VRSRA_N    : SInst<"vrsra_n", "dddi", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VQSHL_N    : SInst<"vqshl_n", "ddi", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl">;
def VQSHLU_N   : SInst<"vqshlu_n", "udi", "csilQcQsQiQl">;
def VSHRN_N    : IInst<"vshrn_n", "hki", "silUsUiUl">;
def VQSHRUN_N  : SInst<"vqshrun_n", "eki", "sil">;
def VQRSHRUN_N : SInst<"vqrshrun_n", "eki", "sil">;
def VQSHRN_N   : SInst<"vqshrn_n", "hki", "silUsUiUl">;
def VRSHRN_N   : IInst<"vrshrn_n", "hki", "silUsUiUl">;
def VQRSHRN_N  : SInst<"vqrshrn_n", "hki", "silUsUiUl">;
def VSHLL_N    : SInst<"vshll_n", "wdi", "csiUcUsUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.13 Shifts with insert
def VSRI_N : WInst<"vsri_n", "dddi",
                   "csilUcUsUiUlPcPsQcQsQiQlQUcQUsQUiQUlQPcQPs">;
def VSLI_N : WInst<"vsli_n", "dddi",
                   "csilUcUsUiUlPcPsQcQsQiQlQUcQUsQUiQUlQPcQPs">;
}

////////////////////////////////////////////////////////////////////////////////
// E.3.14 Loads and stores of a single vector
def VLD1      : WInst<"vld1", "dc",
                      "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD1_LANE : WInst<"vld1_lane", "dcdi",
                      "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD1_DUP  : WInst<"vld1_dup", "dc",
                      "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST1      : WInst<"vst1", "vpd",
                      "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST1_LANE : WInst<"vst1_lane", "vpdi",
                      "QUcQUsQUiQUlQcQsQiQlQhQfQPcQPsUcUsUiUlcsilhfPcPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.15 Loads and stores of an N-element structure
def VLD2 : WInst<"vld2", "2c", "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD3 : WInst<"vld3", "3c", "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD4 : WInst<"vld4", "4c", "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VLD2_DUP  : WInst<"vld2_dup", "2c", "UcUsUiUlcsilhfPcPs">;
def VLD3_DUP  : WInst<"vld3_dup", "3c", "UcUsUiUlcsilhfPcPs">;
def VLD4_DUP  : WInst<"vld4_dup", "4c", "UcUsUiUlcsilhfPcPs">;
def VLD2_LANE : WInst<"vld2_lane", "2c2i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VLD3_LANE : WInst<"vld3_lane", "3c3i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VLD4_LANE : WInst<"vld4_lane", "4c4i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VST2 : WInst<"vst2", "vp2", "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST3 : WInst<"vst3", "vp3", "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST4 : WInst<"vst4", "vp4", "QUcQUsQUiQcQsQiQhQfQPcQPsUcUsUiUlcsilhfPcPs">;
def VST2_LANE : WInst<"vst2_lane", "vp2i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VST3_LANE : WInst<"vst3_lane", "vp3i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;
def VST4_LANE : WInst<"vst4_lane", "vp4i", "QUsQUiQsQiQhQfQPsUcUsUicsihfPcPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.16 Extract lanes from a vector
let InstName = "vmov" in
def VGET_LANE : IInst<"vget_lane", "sdi",
                      "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.17 Set lanes within a vector
let InstName = "vmov" in
def VSET_LANE : IInst<"vset_lane", "dsdi",
                      "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl">;

////////////////////////////////////////////////////////////////////////////////
// E.3.18 Initialize a vector from bit pattern
def VCREATE : NoTestOpInst<"vcreate", "dl", "csihfUcUsUiUlPcPsl", OP_CAST>;

////////////////////////////////////////////////////////////////////////////////
// E.3.19 Set all lanes to same value
let InstName = "vmov" in {
def VDUP_N   : WOpInst<"vdup_n", "ds",
                       "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl", OP_DUP>;
def VMOV_N   : WOpInst<"vmov_n", "ds",
                       "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl", OP_DUP>;
}
let InstName = "" in
def VDUP_LANE: WOpInst<"vdup_lane", "dgi",
                       "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUl",
                       OP_DUP_LN>;

////////////////////////////////////////////////////////////////////////////////
// E.3.20 Combining vectors
def VCOMBINE : NoTestOpInst<"vcombine", "kdd", "csilhfUcUsUiUlPcPs", OP_CONC>;

////////////////////////////////////////////////////////////////////////////////
// E.3.21 Splitting vectors
let InstName = "vmov" in {
def VGET_HIGH : NoTestOpInst<"vget_high", "dk", "csilhfUcUsUiUlPcPs", OP_HI>;
def VGET_LOW  : NoTestOpInst<"vget_low", "dk", "csilhfUcUsUiUlPcPs", OP_LO>;
}

////////////////////////////////////////////////////////////////////////////////
// E.3.22 Converting vectors
def VCVT_S32     : SInst<"vcvt_s32", "xd",  "fQf">;
def VCVT_U32     : SInst<"vcvt_u32", "ud",  "fQf">;
def VCVT_F16     : SInst<"vcvt_f16", "hk",  "f">;
def VCVT_F32     : SInst<"vcvt_f32", "fd",  "iUiQiQUi">;
def VCVT_F32_F16 : SInst<"vcvt_f32_f16", "fd",  "h">;
let isVCVT_N = 1 in {
def VCVT_N_S32   : SInst<"vcvt_n_s32", "xdi", "fQf">;
def VCVT_N_U32   : SInst<"vcvt_n_u32", "udi", "fQf">;
def VCVT_N_F32   : SInst<"vcvt_n_f32", "fdi", "iUiQiQUi">;
}
def VMOVN        : IInst<"vmovn", "hk",  "silUsUiUl">;
def VMOVL        : SInst<"vmovl", "wd",  "csiUcUsUi">;
def VQMOVN       : SInst<"vqmovn", "hk",  "silUsUiUl">;
def VQMOVUN      : SInst<"vqmovun", "ek",  "sil">;

////////////////////////////////////////////////////////////////////////////////
// E.3.23-24 Table lookup, Extended table lookup
let InstName = "vtbl" in {
def VTBL1 : WInst<"vtbl1", "ddt",  "UccPc">;
def VTBL2 : WInst<"vtbl2", "d2t",  "UccPc">;
def VTBL3 : WInst<"vtbl3", "d3t",  "UccPc">;
def VTBL4 : WInst<"vtbl4", "d4t",  "UccPc">;
}
let InstName = "vtbx" in {
def VTBX1 : WInst<"vtbx1", "dddt", "UccPc">;
def VTBX2 : WInst<"vtbx2", "dd2t", "UccPc">;
def VTBX3 : WInst<"vtbx3", "dd3t", "UccPc">;
def VTBX4 : WInst<"vtbx4", "dd4t", "UccPc">;
}

////////////////////////////////////////////////////////////////////////////////
// E.3.25 Operations with a scalar value
def VMLA_LANE     : IOpInst<"vmla_lane", "dddgi",
                            "siUsUifQsQiQUsQUiQf", OP_MLA_LN>;
def VMLAL_LANE    : SOpInst<"vmlal_lane", "wwddi", "siUsUi", OP_MLAL_LN>;
def VQDMLAL_LANE  : SOpInst<"vqdmlal_lane", "wwddi", "si", OP_QDMLAL_LN>;
def VMLS_LANE     : IOpInst<"vmls_lane", "dddgi",
                            "siUsUifQsQiQUsQUiQf", OP_MLS_LN>;
def VMLSL_LANE    : SOpInst<"vmlsl_lane", "wwddi", "siUsUi", OP_MLSL_LN>;
def VQDMLSL_LANE  : SOpInst<"vqdmlsl_lane", "wwddi", "si", OP_QDMLSL_LN>;
def VMUL_N        : IOpInst<"vmul_n", "dds", "sifUsUiQsQiQfQUsQUi", OP_MUL_N>;
def VMUL_LANE     : IOpInst<"vmul_lane", "ddgi",
                            "sifUsUiQsQiQfQUsQUi", OP_MUL_LN>;
def VMULL_N       : SInst<"vmull_n", "wda", "siUsUi">;
def VMULL_LANE    : SOpInst<"vmull_lane", "wddi", "siUsUi", OP_MULL_LN>;
def VQDMULL_N     : SInst<"vqdmull_n", "wda", "si">;
def VQDMULL_LANE  : SOpInst<"vqdmull_lane", "wddi", "si", OP_QDMULL_LN>;
def VQDMULH_N     : SInst<"vqdmulh_n", "dda", "siQsQi">;
def VQDMULH_LANE  : SOpInst<"vqdmulh_lane", "ddgi", "siQsQi", OP_QDMULH_LN>;
def VQRDMULH_N    : SInst<"vqrdmulh_n", "dda", "siQsQi">;
def VQRDMULH_LANE : SOpInst<"vqrdmulh_lane", "ddgi", "siQsQi", OP_QRDMULH_LN>;
def VMLA_N        : IOpInst<"vmla_n", "ddda", "siUsUifQsQiQUsQUiQf", OP_MLA_N>;
def VMLAL_N       : SOpInst<"vmlal_n", "wwda", "siUsUi", OP_MLAL_N>;
def VQDMLAL_N     : SInst<"vqdmlal_n", "wwda", "si">;
def VMLS_N        : IOpInst<"vmls_n", "ddds", "siUsUifQsQiQUsQUiQf", OP_MLS_N>;
def VMLSL_N       : SOpInst<"vmlsl_n", "wwda", "siUsUi", OP_MLSL_N>;
def VQDMLSL_N     : SInst<"vqdmlsl_n", "wwda", "si">;

////////////////////////////////////////////////////////////////////////////////
// E.3.26 Vector Extract
def VEXT : WInst<"vext", "dddi",
                 "cUcPcsUsPsiUilUlfQcQUcQPcQsQUsQPsQiQUiQlQUlQf">;

////////////////////////////////////////////////////////////////////////////////
// E.3.27 Reverse vector elements
def VREV64 : WOpInst<"vrev64", "dd", "csiUcUsUiPcPsfQcQsQiQUcQUsQUiQPcQPsQf",
                  OP_REV64>;
def VREV32 : WOpInst<"vrev32", "dd", "csUcUsPcPsQcQsQUcQUsQPcQPs", OP_REV32>;
def VREV16 : WOpInst<"vrev16", "dd", "cUcPcQcQUcQPc", OP_REV16>;

////////////////////////////////////////////////////////////////////////////////
// E.3.28 Other single operand arithmetic
def VABS    : SInst<"vabs", "dd", "csifQcQsQiQf">;
def VQABS   : SInst<"vqabs", "dd", "csiQcQsQi">;
def VNEG    : SOpInst<"vneg", "dd", "csifQcQsQiQf", OP_NEG>;
def VQNEG   : SInst<"vqneg", "dd", "csiQcQsQi">;
def VCLS    : SInst<"vcls", "dd", "csiQcQsQi">;
def VCLZ    : IInst<"vclz", "dd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VCNT    : WInst<"vcnt", "dd", "UccPcQUcQcQPc">;
def VRECPE  : SInst<"vrecpe", "dd", "fUiQfQUi">;
def VRSQRTE : SInst<"vrsqrte", "dd", "fUiQfQUi">;

////////////////////////////////////////////////////////////////////////////////
// E.3.29 Logical operations
def VMVN : LOpInst<"vmvn", "dd", "csiUcUsUiPcQcQsQiQUcQUsQUiQPc", OP_NOT>;
def VAND : LOpInst<"vand", "ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_AND>;
def VORR : LOpInst<"vorr", "ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_OR>;
def VEOR : LOpInst<"veor", "ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_XOR>;
def VBIC : LOpInst<"vbic", "ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_ANDN>;
def VORN : LOpInst<"vorn", "ddd", "csilUcUsUiUlQcQsQiQlQUcQUsQUiQUl", OP_ORN>;
let isHiddenLInst = 1 in
def VBSL : SInst<"vbsl", "dudd",
                "csilUcUsUiUlfPcPsQcQsQiQlQUcQUsQUiQUlQfQPcQPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.30 Transposition operations
def VTRN : WInst<"vtrn", "2dd", "csiUcUsUifPcPsQcQsQiQUcQUsQUiQfQPcQPs">;
def VZIP : WInst<"vzip", "2dd", "csiUcUsUifPcPsQcQsQiQUcQUsQUiQfQPcQPs">;
def VUZP : WInst<"vuzp", "2dd", "csiUcUsUifPcPsQcQsQiQUcQUsQUiQfQPcQPs">;

////////////////////////////////////////////////////////////////////////////////
// E.3.31 Vector reinterpret cast operations
def VREINTERPRET
  : NoTestOpInst<"vreinterpret", "dd",
         "csilUcUsUiUlhfPcPsQcQsQiQlQUcQUsQUiQUlQhQfQPcQPs", OP_REINT>;

////////////////////////////////////////////////////////////////////////////////
// Vector fused multiply-add operations

def VFMA : SInst<"vfma", "dddd", "fQf">;

////////////////////////////////////////////////////////////////////////////////
// AArch64 Intrinsics

let isA64 = 1 in {

////////////////////////////////////////////////////////////////////////////////
// Load/Store
// With additional QUl, Ql, Qd type.
def LD1 : WInst<"vld1", "dc",
                "QUcQUsQUiQUlQcQsQiQlQhQfQdQPcQPsUcUsUiUlcsilhfdPcPs">;
def LD2 : WInst<"vld2", "2c",
                "QUcQUsQUiQUlQcQsQiQlQhQfQdQPcQPsUcUsUiUlcsilhfdPcPs">;
def LD3 : WInst<"vld3", "3c",
                "QUcQUsQUiQUlQcQsQiQlQhQfQdQPcQPsUcUsUiUlcsilhfdPcPs">;
def LD4 : WInst<"vld4", "4c",
                "QUcQUsQUiQUlQcQsQiQlQhQfQdQPcQPsUcUsUiUlcsilhfdPcPs">;
def ST1 : WInst<"vst1", "vpd",
                "QUcQUsQUiQUlQcQsQiQlQhQfQdQPcQPsUcUsUiUlcsilhfdPcPs">;
def ST2 : WInst<"vst2", "vp2",
                "QUcQUsQUiQUlQcQsQiQlQhQfQdQPcQPsUcUsUiUlcsilhfdPcPs">;
def ST3 : WInst<"vst3", "vp3",
                "QUcQUsQUiQUlQcQsQiQlQhQfQdQPcQPsUcUsUiUlcsilhfdPcPs">;
def ST4 : WInst<"vst4", "vp4",
                "QUcQUsQUiQUlQcQsQiQlQhQfQdQPcQPsUcUsUiUlcsilhfdPcPs">;

////////////////////////////////////////////////////////////////////////////////
// Addition
// With additional Qd type.
def ADD : IOpInst<"vadd", "ddd", "csilfUcUsUiUlQcQsQiQlQfQUcQUsQUiQUlQd", OP_ADD>;

////////////////////////////////////////////////////////////////////////////////
// Subtraction
// With additional Qd type.
def SUB : IOpInst<"vsub", "ddd", "csilfUcUsUiUlQcQsQiQlQfQUcQUsQUiQUlQd", OP_SUB>;

////////////////////////////////////////////////////////////////////////////////
// Multiplication
// With additional Qd type.
def MUL     : IOpInst<"vmul", "ddd", "csifUcUsUiQcQsQiQfQUcQUsQUiQd", OP_MUL>;
def MLA     : IOpInst<"vmla", "dddd", "csifUcUsUiQcQsQiQfQUcQUsQUiQd", OP_MLA>;
def MLS     : IOpInst<"vmls", "dddd", "csifUcUsUiQcQsQiQfQUcQUsQUiQd", OP_MLS>;

////////////////////////////////////////////////////////////////////////////////
// Multiplication Extended
def MULX : SInst<"vmulx", "ddd", "fdQfQd">;

////////////////////////////////////////////////////////////////////////////////
// Division
def FDIV : IOpInst<"vdiv", "ddd",  "fQfQd", OP_DIV>;

////////////////////////////////////////////////////////////////////////////////
// Vector fused multiply-add operations
// With additional Qd type.
def FMLA : SInst<"vfma", "dddd", "fQfQd">;
def FMLS : SInst<"vfms", "dddd", "fQfQd">;

////////////////////////////////////////////////////////////////////////////////
// Logical operations
// With additional Qd type.
def BSL : SInst<"vbsl", "dudd", "csilUcUsUiUlfPcPsQcQsQiQlQUcQUsQUiQUlQfQPcQPsQd">;

////////////////////////////////////////////////////////////////////////////////
// Absolute Difference
// With additional Qd type.
def ABD  : SInst<"vabd", "ddd",  "csiUcUsUifQcQsQiQUcQUsQUiQfQd">;

////////////////////////////////////////////////////////////////////////////////
// Reciprocal/Sqrt
// With additional Qd type.
def FRECPS  : IInst<"vrecps", "ddd", "fQfQd">;
def FRSQRTS : IInst<"vrsqrts", "ddd", "fQfQd">;

////////////////////////////////////////////////////////////////////////////////
// Comparison
// With additional Qd type.
def FCAGE : IInst<"vcage", "udd", "fQfQd">;
def FCAGT : IInst<"vcagt", "udd", "fQfQd">;
def FCALE : IInst<"vcale", "udd", "fQfQd">;
def FCALT : IInst<"vcalt", "udd", "fQfQd">;
// With additional Ql, QUl, Qd types.
def CMTST  : WInst<"vtst", "udd", "csiUcUsUiPcPsQcQsQiQlQUcQUsQUiQUlQPcQPs">;
def CFMEQ  : SOpInst<"vceq", "udd",
                  "csifUcUsUiPcQcQsQiQlQfQUcQUsQUiQUlQPcQd", OP_EQ>;
def CFMGE  : SOpInst<"vcge", "udd", "csifUcUsUiQcQsQiQlQfQUcQUsQUiQUlQd", OP_GE>;
def CFMLE  : SOpInst<"vcle", "udd", "csifUcUsUiQcQsQiQlQfQUcQUsQUiQUlQd", OP_LE>;
def CFMGT  : SOpInst<"vcgt", "udd", "csifUcUsUiQcQsQiQlQfQUcQUsQUiQUlQd", OP_GT>;
def CFMLT  : SOpInst<"vclt", "udd", "csifUcUsUiQcQsQiQlQfQUcQUsQUiQUlQd", OP_LT>;

////////////////////////////////////////////////////////////////////////////////
// Max/Min Integer
// With additional Qd type.
def MAX : SInst<"vmax", "ddd", "csiUcUsUifQcQsQiQUcQUsQUiQfQd">;
def MIN : SInst<"vmin", "ddd", "csiUcUsUifQcQsQiQUcQUsQUiQfQd">;

////////////////////////////////////////////////////////////////////////////////
// MaxNum/MinNum Floating Point
def FMAXNM : SInst<"vmaxnm", "ddd", "fQfQd">;
def FMINNM : SInst<"vminnm", "ddd", "fQfQd">;

////////////////////////////////////////////////////////////////////////////////
// Pairwise Max/Min
// With additional Qc Qs Qi QUc QUs QUi Qf Qd types.
def MAXP : SInst<"vpmax", "ddd", "csiUcUsUifQcQsQiQUcQUsQUiQfQd">;
def MINP : SInst<"vpmin", "ddd", "csiUcUsUifQcQsQiQUcQUsQUiQfQd">;

////////////////////////////////////////////////////////////////////////////////
// Pairwise MaxNum/MinNum Floating Point
def FMAXNMP : SInst<"vpmaxnm", "ddd", "fQfQd">;
def FMINNMP : SInst<"vpminnm", "ddd", "fQfQd">;

////////////////////////////////////////////////////////////////////////////////
// Pairwise Addition
// With additional Qc Qs Qi QUc QUs QUi Qf Qd types.
def ADDP  : IInst<"vpadd", "ddd", "csiUcUsUifQcQsQiQUcQUsQUiQfQd">;

////////////////////////////////////////////////////////////////////////////////
// Shifts by constant
let isShift = 1 in {
// Left shift long high
def SHLL_HIGH_N    : SOpInst<"vshll_high_n", "ndi", "HcHsHiHUcHUsHUi",
                             OP_LONG_HI>;

// Right shift narrow high
def SHRN_HIGH_N    : IOpInst<"vshrn_high_n", "hmdi",
                             "HsHiHlHUsHUiHUl", OP_NARROW_HI>;
def QSHRUN_HIGH_N  : SOpInst<"vqshrun_high_n", "hmdi",
                             "HsHiHl", OP_NARROW_HI>;
def RSHRN_HIGH_N   : IOpInst<"vrshrn_high_n", "hmdi",
                             "HsHiHlHUsHUiHUl", OP_NARROW_HI>;
def QRSHRUN_HIGH_N : SOpInst<"vqrshrun_high_n", "hmdi",
                             "HsHiHl", OP_NARROW_HI>;
def QSHRN_HIGH_N   : SOpInst<"vqshrn_high_n", "hmdi", 
                             "HsHiHlHUsHUiHUl", OP_NARROW_HI>;
def QRSHRN_HIGH_N  : SOpInst<"vqrshrn_high_n", "hmdi", 
                             "HsHiHlHUsHUiHUl", OP_NARROW_HI>;
}

////////////////////////////////////////////////////////////////////////////////
// Converting vectors
def VMOVL_HIGH   : SOpInst<"vmovl_high", "nd", "HcHsHiHUcHUsHUi", OP_MOVL_HI>;

let isVCVT_N = 1 in {
def CVTF_N_F64   : SInst<"vcvt_n_f64", "fdi", "QlQUl">;
def FCVTZS_N_S64 : SInst<"vcvt_n_s64", "xdi", "Qd">;
def FCVTZS_N_U64 : SInst<"vcvt_n_u64", "udi", "Qd">;
}

////////////////////////////////////////////////////////////////////////////////
// 3VDiff class using high 64-bit in operands
def VADDL_HIGH   : SOpInst<"vaddl_high", "wkk", "csiUcUsUi", OP_ADDLHi>;
def VADDW_HIGH   : SOpInst<"vaddw_high", "wwk", "csiUcUsUi", OP_ADDWHi>;
def VSUBL_HIGH   : SOpInst<"vsubl_high", "wkk", "csiUcUsUi", OP_SUBLHi>;
def VSUBW_HIGH   : SOpInst<"vsubw_high", "wwk", "csiUcUsUi", OP_SUBWHi>;

def VABDL_HIGH   : SOpInst<"vabdl_high", "wkk",  "csiUcUsUi", OP_ABDLHi>;
def VABAL_HIGH   : SOpInst<"vabal_high", "wwkk", "csiUcUsUi", OP_ABALHi>;

def VMULL_HIGH   : SOpInst<"vmull_high", "wkk", "csiUcUsUiPc", OP_MULLHi>;
def VMLAL_HIGH   : SOpInst<"vmlal_high", "wwkk", "csiUcUsUi", OP_MLALHi>;
def VMLSL_HIGH   : SOpInst<"vmlsl_high", "wwkk", "csiUcUsUi", OP_MLSLHi>;

def VADDHN_HIGH  : SOpInst<"vaddhn_high", "qhkk", "silUsUiUl", OP_ADDHNHi>;
def VRADDHN_HIGH : SOpInst<"vraddhn_high", "qhkk", "silUsUiUl", OP_RADDHNHi>;
def VSUBHN_HIGH  : SOpInst<"vsubhn_high", "qhkk", "silUsUiUl", OP_SUBHNHi>;
def VRSUBHN_HIGH : SOpInst<"vrsubhn_high", "qhkk", "silUsUiUl", OP_RSUBHNHi>;

def VQDMULL_HIGH : SOpInst<"vqdmull_high", "wkk", "si", OP_QDMULLHi>;
def VQDMLAL_HIGH : SOpInst<"vqdmlal_high", "wwkk", "si", OP_QDMLALHi>;
def VQDMLSL_HIGH : SOpInst<"vqdmlsl_high", "wwkk", "si", OP_QDMLSLHi>;

////////////////////////////////////////////////////////////////////////////////
// Extract or insert element from vector
def GET_LANE : IInst<"vget_lane", "sdi",
                        "csilPcPsUcUsUiUlQcQsQiQlQUcQUsQUiQUlPcPsQPcQPs">;
def SET_LANE : IInst<"vset_lane", "dsdi",
                 "csilPcPsUiUlUcUsUiUlQcQsQiQlQUcQUsQUiQUlPcPsQPcQPshfdQhQfQd">;
def COPY_LANE : IOpInst<"vcopy_lane", "ddidi",
                     "csilPcPsUcUsUiUlQcQsQiQlQUcQUsQUiQUlPcPsQPcQPs", OP_COPY>;
def COPY_LANEQ : IOpInst<"vcopy_laneq", "ggidi",
                     "QcQsQiQlQUcQUsQUiQUlQPcQPs", OP_COPY>;

////////////////////////////////////////////////////////////////////////////////
// Set all lanes to same value
def VDUP_LANE1: WOpInst<"vdup_lane", "dgi",
                        "csilPcPsUcUsUiUlhfdQcQsQiQlQPcQPsQUcQUsQUiQUlQhQfQd",
                        OP_DUP_LN>;
def VDUP_LANE2: WOpInst<"vdup_laneq", "dki",
                        "csilPcPsUcUsUiUlhfdQcQsQiQlQPcQPsQUcQUsQUiQUlQhQfQd",
                        OP_DUP_LN>;
def DUP_N   : WOpInst<"vdup_n", "ds",
                       "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUldQd",
                       OP_DUP>;
def MOV_N   : WOpInst<"vmov_n", "ds",
                       "UcUsUicsiPcPsfQUcQUsQUiQcQsQiQPcQPsQflUlQlQUldQd",
                       OP_DUP>;
////////////////////////////////////////////////////////////////////////////////
//Initialize a vector from bit pattern
def CREATE : NoTestOpInst<"vcreate", "dl", "csihfdUcUsUiUlPcPsl", OP_CAST>;

////////////////////////////////////////////////////////////////////////////////

def VMLA_LANEQ   : IOpInst<"vmla_laneq", "dddji",
                           "siUsUifQsQiQUsQUiQf", OP_MLA_LN>;
def VMLS_LANEQ   : IOpInst<"vmls_laneq", "dddji",
                           "siUsUifQsQiQUsQUiQf", OP_MLS_LN>;

def VFMA_LANE    : IInst<"vfma_lane", "dddgi", "fdQfQd">;
def VFMA_LANEQ   : IInst<"vfma_laneq", "dddji", "fdQfQd">;
def VFMS_LANE    : IOpInst<"vfms_lane", "dddgi", "fdQfQd", OP_FMS_LN>;
def VFMS_LANEQ   : IOpInst<"vfms_laneq", "dddji", "fdQfQd", OP_FMS_LNQ>;

def VMLAL_LANEQ  : SOpInst<"vmlal_laneq", "wwdki", "siUsUi", OP_MLAL_LN>;
def VMLAL_HIGH_LANE   : SOpInst<"vmlal_high_lane", "wwkdi", "siUsUi",
                                OP_MLALHi_LN>;
def VMLAL_HIGH_LANEQ  : SOpInst<"vmlal_high_laneq", "wwkki", "siUsUi",
                                OP_MLALHi_LN>;
def VMLSL_LANEQ  : SOpInst<"vmlsl_laneq", "wwdki", "siUsUi", OP_MLSL_LN>;
def VMLSL_HIGH_LANE   : SOpInst<"vmlsl_high_lane", "wwkdi", "siUsUi",
                                OP_MLSLHi_LN>;
def VMLSL_HIGH_LANEQ  : SOpInst<"vmlsl_high_laneq", "wwkki", "siUsUi",
                                OP_MLSLHi_LN>;

def VQDMLAL_LANEQ  : SOpInst<"vqdmlal_laneq", "wwdki", "si", OP_QDMLAL_LN>;
def VQDMLAL_HIGH_LANE   : SOpInst<"vqdmlal_high_lane", "wwkdi", "si",
                                OP_QDMLALHi_LN>;
def VQDMLAL_HIGH_LANEQ  : SOpInst<"vqdmlal_high_laneq", "wwkki", "si",
                                OP_QDMLALHi_LN>;
def VQDMLSL_LANEQ  : SOpInst<"vqdmlsl_laneq", "wwdki", "si", OP_QDMLSL_LN>;
def VQDMLSL_HIGH_LANE   : SOpInst<"vqdmlsl_high_lane", "wwkdi", "si",
                                OP_QDMLSLHi_LN>;
def VQDMLSL_HIGH_LANEQ  : SOpInst<"vqdmlsl_high_laneq", "wwkki", "si",
                                OP_QDMLSLHi_LN>;

// Newly add double parameter for vmul_lane in aarch64
def VMUL_LANE_A64 : IOpInst<"vmul_lane", "ddgi", "dQd", OP_MUL_LN>;

def VMUL_LANEQ   : IOpInst<"vmul_laneq", "ddji",
                           "sifdUsUiQsQiQfQUsQUiQfQd", OP_MUL_LN>;
def VMULL_LANEQ  : SOpInst<"vmull_laneq", "wdki", "siUsUi", OP_MULL_LN>;
def VMULL_HIGH_LANE   : SOpInst<"vmull_high_lane", "wkdi", "siUsUi",
                                OP_MULLHi_LN>;
def VMULL_HIGH_LANEQ  : SOpInst<"vmull_high_laneq", "wkki", "siUsUi",
                                OP_MULLHi_LN>;

def VQDMULL_LANEQ  : SOpInst<"vqdmull_laneq", "wdki", "si", OP_QDMULL_LN>;
def VQDMULL_HIGH_LANE   : SOpInst<"vqdmull_high_lane", "wkdi", "si",
                                  OP_QDMULLHi_LN>;
def VQDMULL_HIGH_LANEQ  : SOpInst<"vqdmull_high_laneq", "wkki", "si",
                                  OP_QDMULLHi_LN>;

def VQDMULH_LANEQ  : SOpInst<"vqdmulh_laneq", "ddji", "siQsQi", OP_QDMULH_LN>;
def VQRDMULH_LANEQ : SOpInst<"vqrdmulh_laneq", "ddji", "siQsQi", OP_QRDMULH_LN>;

def VMULX_LANE : IOpInst<"vmulx_lane", "ddgi", "fdQfQd", OP_MULX_LN>;
def VMULX_LANEQ : IOpInst<"vmulx_laneq", "ddji", "fdQfQd", OP_MULX_LN>;

////////////////////////////////////////////////////////////////////////////////
// Across vectors class
def VADDLV  : SInst<"vaddlv", "rd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def VMAXV   : SInst<"vmaxv", "sd", "csiUcUsUiQcQsQiQUcQUsQUiQf">;
def VMINV   : SInst<"vminv", "sd", "csiUcUsUiQcQsQiQUcQUsQUiQf">;
def VADDV   : SInst<"vaddv", "sd", "csiUcUsUiQcQsQiQUcQUsQUi">;
def FMAXNMV : SInst<"vmaxnmv", "sd", "Qf">;
def FMINNMV : SInst<"vminnmv", "sd", "Qf">;

////////////////////////////////////////////////////////////////////////////////
// Scalar Arithmetic

// Scalar Addition
def SCALAR_ADD : SInst<"vadd", "sss",  "SlSUl">;
// Scalar  Saturating Add
def SCALAR_QADD   : SInst<"vqadd", "sss", "ScSsSiSlSUcSUsSUiSUl">;

// Scalar Subtraction
def SCALAR_SUB : SInst<"vsub", "sss",  "SlSUl">;
// Scalar  Saturating Sub
def SCALAR_QSUB   : SInst<"vqsub", "sss", "ScSsSiSlSUcSUsSUiSUl">;

////////////////////////////////////////////////////////////////////////////////
// Scalar Shift
// Scalar Shift Left
def SCALAR_SHL: SInst<"vshl", "sss", "SlSUl">;
// Scalar Saturating Shift Left
def SCALAR_QSHL: SInst<"vqshl", "sss", "ScSsSiSlSUcSUsSUiSUl">;
// Scalar Saturating Rounding Shift Left
def SCALAR_QRSHL: SInst<"vqrshl", "sss", "ScSsSiSlSUcSUsSUiSUl">;
// Scalar Shift Rouding Left
def SCALAR_RSHL: SInst<"vrshl", "sss", "SlSUl">;

////////////////////////////////////////////////////////////////////////////////
// Scalar Reduce Pairwise Addition (Scalar and Floating Point)
def SCALAR_ADDP  : SInst<"vpadd", "sd", "SfSHlSHd">;

////////////////////////////////////////////////////////////////////////////////
// Scalar Reduce Floating Point Pairwise Max/Min
def SCALAR_FMAXP : SInst<"vpmax", "sd", "SfSQd">;

def SCALAR_FMINP : SInst<"vpmin", "sd", "SfSQd">;

////////////////////////////////////////////////////////////////////////////////
// Scalar Reduce Floating Point Pairwise maxNum/minNum
def SCALAR_FMAXNMP : SInst<"vpmaxnm", "sd", "SfSQd">;
def SCALAR_FMINNMP : SInst<"vpminnm", "sd", "SfSQd">;

////////////////////////////////////////////////////////////////////////////////
// Scalar Integer Saturating Doubling Multiply Half High
def SCALAR_SQDMULH : SInst<"vqdmulh", "sss", "SsSi">;

////////////////////////////////////////////////////////////////////////////////
// Scalar Integer Saturating Rounding Doubling Multiply Half High
def SCALAR_SQRDMULH : SInst<"vqrdmulh", "sss", "SsSi">;

////////////////////////////////////////////////////////////////////////////////
// Scalar Floating-point Multiply Extended
def SCALAR_FMULX : IInst<"vmulx", "sss", "SfSd">;

////////////////////////////////////////////////////////////////////////////////
// Scalar Floating-point Reciprocal Step
def SCALAR_FRECPS : IInst<"vrecps", "sss", "SfSd">;

////////////////////////////////////////////////////////////////////////////////
// Scalar Floating-point Reciprocal Square Root Step
def SCALAR_FRSQRTS : IInst<"vrsqrts", "sss", "SfSd">;

////////////////////////////////////////////////////////////////////////////////
// Scalar Signed Integer Convert To Floating-point
def SCALAR_SCVTFS : SInst<"vcvt_f32", "ys", "Si">;
def SCALAR_SCVTFD : SInst<"vcvt_f64", "os", "Sl">;

////////////////////////////////////////////////////////////////////////////////
// Scalar Unsigned Integer Convert To Floating-point
def SCALAR_UCVTFS : SInst<"vcvt_f32", "ys", "SUi">;
def SCALAR_UCVTFD : SInst<"vcvt_f64", "os", "SUl">;

////////////////////////////////////////////////////////////////////////////////
// Scalar Floating-point Reciprocal Estimate
def SCALAR_FRECPE : IInst<"vrecpe", "ss", "SfSd">;

////////////////////////////////////////////////////////////////////////////////
// Scalar Floating-point Reciprocal Exponent
def SCALAR_FRECPX : IInst<"vrecpx", "ss", "SfSd">;

////////////////////////////////////////////////////////////////////////////////
// Scalar Floating-point Reciprocal Square Root Estimate
def SCALAR_FRSQRTE : IInst<"vrsqrte", "ss", "SfSd">;

}
