//===- NVPTXIntrinsics.td - PTX Intrinsics Instructions -------*- tblgen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def immFloat0 : PatLeaf<(fpimm), [{
    float f = (float)N->getValueAPF().convertToFloat();
    return (f==0.0f);
}]>;

def immFloat1 : PatLeaf<(fpimm), [{
    float f = (float)N->getValueAPF().convertToFloat();
    return (f==1.0f);
}]>;

def immDouble0 : PatLeaf<(fpimm), [{
    double d = (double)N->getValueAPF().convertToDouble();
    return (d==0.0);
}]>;

def immDouble1 : PatLeaf<(fpimm), [{
    double d = (double)N->getValueAPF().convertToDouble();
    return (d==1.0);
}]>;



//-----------------------------------
// Synchronization Functions
//-----------------------------------
def INT_CUDA_SYNCTHREADS : NVPTXInst<(outs), (ins),
                  "bar.sync \t0;",
      [(int_cuda_syncthreads)]>;
def INT_BARRIER0 : NVPTXInst<(outs), (ins),
                  "bar.sync \t0;",
      [(int_nvvm_barrier0)]>;
def INT_BARRIER0_POPC : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$pred),
  !strconcat("{{ \n\t",
      !strconcat(".reg .pred \t%p1; \n\t",
      !strconcat("setp.ne.u32 \t%p1, $pred, 0; \n\t",
      !strconcat("bar.red.popc.u32 \t$dst, 0, %p1; \n\t",
        !strconcat("}}", ""))))),
      [(set Int32Regs:$dst, (int_nvvm_barrier0_popc Int32Regs:$pred))]>;
def INT_BARRIER0_AND : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$pred),
  !strconcat("{{ \n\t",
      !strconcat(".reg .pred \t%p1; \n\t",
      !strconcat(".reg .pred \t%p2; \n\t",
      !strconcat("setp.ne.u32 \t%p1, $pred, 0; \n\t",
      !strconcat("bar.red.and.pred \t%p2, 0, %p1; \n\t",
      !strconcat("selp.u32 \t$dst, 1, 0, %p2; \n\t",
        !strconcat("}}", ""))))))),
      [(set Int32Regs:$dst, (int_nvvm_barrier0_and Int32Regs:$pred))]>;
def INT_BARRIER0_OR : NVPTXInst<(outs Int32Regs:$dst), (ins Int32Regs:$pred),
  !strconcat("{{ \n\t",
      !strconcat(".reg .pred \t%p1; \n\t",
      !strconcat(".reg .pred \t%p2; \n\t",
      !strconcat("setp.ne.u32 \t%p1, $pred, 0; \n\t",
      !strconcat("bar.red.or.pred \t%p2, 0, %p1; \n\t",
      !strconcat("selp.u32 \t$dst, 1, 0, %p2; \n\t",
        !strconcat("}}", ""))))))),
      [(set Int32Regs:$dst, (int_nvvm_barrier0_or Int32Regs:$pred))]>;


//-----------------------------------
// Explicit Memory Fence Functions
//-----------------------------------
class MEMBAR<string StrOp, Intrinsic IntOP> :
              NVPTXInst<(outs), (ins),
            StrOp, [(IntOP)]>;

def INT_MEMBAR_CTA : MEMBAR<"membar.cta;", int_nvvm_membar_cta>;
def INT_MEMBAR_GL  : MEMBAR<"membar.gl;",  int_nvvm_membar_gl>;
def INT_MEMBAR_SYS : MEMBAR<"membar.sys;", int_nvvm_membar_sys>;


//-----------------------------------
// Math Functions
//-----------------------------------

// Map min(1.0, max(0.0, x)) to sat(x)
// Note that max(0.0, min(x, 1.0)) cannot be mapped to sat(x) because when x is
// NaN
// max(0.0, min(x, 1.0)) is 1.0 while sat(x) is 0.
// Same story for fmax, fmin.

def : Pat<(int_nvvm_fmin_f immFloat1,
            (int_nvvm_fmax_f immFloat0, Float32Regs:$a)),
          (CVT_f32_f32 Float32Regs:$a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_f immFloat1,
            (int_nvvm_fmax_f Float32Regs:$a, immFloat0)),
          (CVT_f32_f32 Float32Regs:$a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_f
            (int_nvvm_fmax_f immFloat0, Float32Regs:$a), immFloat1),
          (CVT_f32_f32 Float32Regs:$a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_f
            (int_nvvm_fmax_f Float32Regs:$a, immFloat0), immFloat1),
          (CVT_f32_f32 Float32Regs:$a, CvtSAT)>;

def : Pat<(int_nvvm_fmin_d immDouble1,
            (int_nvvm_fmax_d immDouble0, Float64Regs:$a)),
          (CVT_f64_f64 Float64Regs:$a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_d immDouble1,
            (int_nvvm_fmax_d Float64Regs:$a, immDouble0)),
          (CVT_f64_f64 Float64Regs:$a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_d
            (int_nvvm_fmax_d immDouble0, Float64Regs:$a), immDouble1),
          (CVT_f64_f64 Float64Regs:$a, CvtSAT)>;
def : Pat<(int_nvvm_fmin_d
            (int_nvvm_fmax_d Float64Regs:$a, immDouble0), immDouble1),
          (CVT_f64_f64 Float64Regs:$a, CvtSAT)>;


// We need a full string for OpcStr here because we need to deal with case like
// INT_PTX_RECIP.
class F_MATH_1<string OpcStr, NVPTXRegClass target_regclass,
  NVPTXRegClass src_regclass, Intrinsic IntOP>
            : NVPTXInst<(outs target_regclass:$dst), (ins src_regclass:$src0),
            OpcStr,
        [(set target_regclass:$dst, (IntOP src_regclass:$src0))]>;

// We need a full string for OpcStr here because we need to deal with the case
// like INT_PTX_NATIVE_POWR_F.
class F_MATH_2<string OpcStr, NVPTXRegClass t_regclass,
  NVPTXRegClass s0_regclass, NVPTXRegClass s1_regclass, Intrinsic IntOP>
            : NVPTXInst<(outs t_regclass:$dst),
              (ins s0_regclass:$src0, s1_regclass:$src1),
            OpcStr,
        [(set t_regclass:$dst, (IntOP s0_regclass:$src0, s1_regclass:$src1))]>;

class F_MATH_3<string OpcStr, NVPTXRegClass t_regclass,
  NVPTXRegClass s0_regclass, NVPTXRegClass s1_regclass,
  NVPTXRegClass s2_regclass, Intrinsic IntOP>
            : NVPTXInst<(outs t_regclass:$dst),
              (ins s0_regclass:$src0, s1_regclass:$src1, s2_regclass:$src2),
            OpcStr,
        [(set t_regclass:$dst,
          (IntOP s0_regclass:$src0, s1_regclass:$src1, s2_regclass:$src2))]>;

//
// MISC
//

def INT_NVVM_CLZ_I : F_MATH_1<"clz.b32 \t$dst, $src0;", Int32Regs, Int32Regs,
  int_nvvm_clz_i>;
def INT_NVVM_CLZ_LL : F_MATH_1<"clz.b64 \t$dst, $src0;", Int32Regs, Int64Regs,
  int_nvvm_clz_ll>;

def INT_NVVM_POPC_I : F_MATH_1<"popc.b32 \t$dst, $src0;", Int32Regs, Int32Regs,
  int_nvvm_popc_i>;
def INT_NVVM_POPC_LL : F_MATH_1<"popc.b64 \t$dst, $src0;", Int32Regs, Int64Regs,
  int_nvvm_popc_ll>;

def INT_NVVM_PRMT : F_MATH_3<"prmt.b32 \t$dst, $src0, $src1, $src2;", Int32Regs,
  Int32Regs, Int32Regs, Int32Regs, int_nvvm_prmt>;

//
// Min Max
//

def INT_NVVM_MIN_I : F_MATH_2<"min.s32 \t$dst, $src0, $src1;", Int32Regs,
  Int32Regs, Int32Regs, int_nvvm_min_i>;
def INT_NVVM_MIN_UI : F_MATH_2<"min.u32 \t$dst, $src0, $src1;", Int32Regs,
  Int32Regs, Int32Regs, int_nvvm_min_ui>;

def INT_NVVM_MIN_LL : F_MATH_2<"min.s64 \t$dst, $src0, $src1;", Int64Regs,
  Int64Regs, Int64Regs, int_nvvm_min_ll>;
def INT_NVVM_MIN_ULL : F_MATH_2<"min.u64 \t$dst, $src0, $src1;", Int64Regs,
  Int64Regs, Int64Regs, int_nvvm_min_ull>;

def INT_NVVM_MAX_I : F_MATH_2<"max.s32 \t$dst, $src0, $src1;", Int32Regs,
  Int32Regs, Int32Regs, int_nvvm_max_i>;
def INT_NVVM_MAX_UI : F_MATH_2<"max.u32 \t$dst, $src0, $src1;", Int32Regs,
  Int32Regs, Int32Regs, int_nvvm_max_ui>;

def INT_NVVM_MAX_LL : F_MATH_2<"max.s64 \t$dst, $src0, $src1;", Int64Regs,
  Int64Regs, Int64Regs, int_nvvm_max_ll>;
def INT_NVVM_MAX_ULL : F_MATH_2<"max.u64 \t$dst, $src0, $src1;", Int64Regs,
  Int64Regs, Int64Regs, int_nvvm_max_ull>;

def INT_NVVM_FMIN_F : F_MATH_2<"min.f32 \t$dst, $src0, $src1;", Float32Regs,
  Float32Regs, Float32Regs, int_nvvm_fmin_f>;
def INT_NVVM_FMIN_FTZ_F : F_MATH_2<"min.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_fmin_ftz_f>;

def INT_NVVM_FMAX_F : F_MATH_2<"max.f32 \t$dst, $src0, $src1;", Float32Regs,
  Float32Regs, Float32Regs, int_nvvm_fmax_f>;
def INT_NVVM_FMAX_FTZ_F : F_MATH_2<"max.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_fmax_ftz_f>;

def INT_NVVM_FMIN_D : F_MATH_2<"min.f64 \t$dst, $src0, $src1;", Float64Regs,
  Float64Regs, Float64Regs, int_nvvm_fmin_d>;
def INT_NVVM_FMAX_D : F_MATH_2<"max.f64 \t$dst, $src0, $src1;", Float64Regs,
  Float64Regs, Float64Regs, int_nvvm_fmax_d>;

//
// Multiplication
//

def INT_NVVM_MULHI_I : F_MATH_2<"mul.hi.s32 \t$dst, $src0, $src1;", Int32Regs,
  Int32Regs, Int32Regs, int_nvvm_mulhi_i>;
def INT_NVVM_MULHI_UI : F_MATH_2<"mul.hi.u32 \t$dst, $src0, $src1;", Int32Regs,
  Int32Regs, Int32Regs, int_nvvm_mulhi_ui>;

def INT_NVVM_MULHI_LL : F_MATH_2<"mul.hi.s64 \t$dst, $src0, $src1;", Int64Regs,
  Int64Regs, Int64Regs, int_nvvm_mulhi_ll>;
def INT_NVVM_MULHI_ULL : F_MATH_2<"mul.hi.u64 \t$dst, $src0, $src1;", Int64Regs,
  Int64Regs, Int64Regs, int_nvvm_mulhi_ull>;

def INT_NVVM_MUL_RN_FTZ_F : F_MATH_2<"mul.rn.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_mul_rn_ftz_f>;
def INT_NVVM_MUL_RN_F : F_MATH_2<"mul.rn.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_mul_rn_f>;
def INT_NVVM_MUL_RZ_FTZ_F : F_MATH_2<"mul.rz.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_mul_rz_ftz_f>;
def INT_NVVM_MUL_RZ_F : F_MATH_2<"mul.rz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_mul_rz_f>;
def INT_NVVM_MUL_RM_FTZ_F : F_MATH_2<"mul.rm.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_mul_rm_ftz_f>;
def INT_NVVM_MUL_RM_F : F_MATH_2<"mul.rm.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_mul_rm_f>;
def INT_NVVM_MUL_RP_FTZ_F : F_MATH_2<"mul.rp.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_mul_rp_ftz_f>;
def INT_NVVM_MUL_RP_F : F_MATH_2<"mul.rp.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_mul_rp_f>;

def INT_NVVM_MUL_RN_D : F_MATH_2<"mul.rn.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_mul_rn_d>;
def INT_NVVM_MUL_RZ_D : F_MATH_2<"mul.rz.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_mul_rz_d>;
def INT_NVVM_MUL_RM_D : F_MATH_2<"mul.rm.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_mul_rm_d>;
def INT_NVVM_MUL_RP_D : F_MATH_2<"mul.rp.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_mul_rp_d>;

def INT_NVVM_MUL24_I : F_MATH_2<"mul24.lo.s32 \t$dst, $src0, $src1;",
  Int32Regs, Int32Regs, Int32Regs, int_nvvm_mul24_i>;
def INT_NVVM_MUL24_UI : F_MATH_2<"mul24.lo.u32 \t$dst, $src0, $src1;",
  Int32Regs, Int32Regs, Int32Regs, int_nvvm_mul24_ui>;

//
// Div
//

def INT_NVVM_DIV_APPROX_FTZ_F
  : F_MATH_2<"div.approx.ftz.f32 \t$dst, $src0, $src1;", Float32Regs,
    Float32Regs, Float32Regs, int_nvvm_div_approx_ftz_f>;
def INT_NVVM_DIV_APPROX_F : F_MATH_2<"div.approx.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_approx_f>;

def INT_NVVM_DIV_RN_FTZ_F : F_MATH_2<"div.rn.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_rn_ftz_f>;
def INT_NVVM_DIV_RN_F     : F_MATH_2<"div.rn.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_rn_f>;
def INT_NVVM_DIV_RZ_FTZ_F : F_MATH_2<"div.rz.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_rz_ftz_f>;
def INT_NVVM_DIV_RZ_F     : F_MATH_2<"div.rz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_rz_f>;
def INT_NVVM_DIV_RM_FTZ_F : F_MATH_2<"div.rm.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_rm_ftz_f>;
def INT_NVVM_DIV_RM_F     : F_MATH_2<"div.rm.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_rm_f>;
def INT_NVVM_DIV_RP_FTZ_F : F_MATH_2<"div.rp.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_rp_ftz_f>;
def INT_NVVM_DIV_RP_F     : F_MATH_2<"div.rp.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_div_rp_f>;

def INT_NVVM_DIV_RN_D : F_MATH_2<"div.rn.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_div_rn_d>;
def INT_NVVM_DIV_RZ_D : F_MATH_2<"div.rz.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_div_rz_d>;
def INT_NVVM_DIV_RM_D : F_MATH_2<"div.rm.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_div_rm_d>;
def INT_NVVM_DIV_RP_D : F_MATH_2<"div.rp.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_div_rp_d>;

//
// Brev
//

def INT_NVVM_BREV32 : F_MATH_1<"brev.b32 \t$dst, $src0;", Int32Regs, Int32Regs,
  int_nvvm_brev32>;
def INT_NVVM_BREV64 : F_MATH_1<"brev.b64 \t$dst, $src0;", Int64Regs, Int64Regs,
  int_nvvm_brev64>;

//
// Sad
//

def INT_NVVM_SAD_I : F_MATH_3<"sad.s32 \t$dst, $src0, $src1, $src2;",
  Int32Regs, Int32Regs, Int32Regs, Int32Regs, int_nvvm_sad_i>;
def INT_NVVM_SAD_UI : F_MATH_3<"sad.u32 \t$dst, $src0, $src1, $src2;",
  Int32Regs, Int32Regs, Int32Regs, Int32Regs, int_nvvm_sad_ui>;

//
// Floor  Ceil
//

def : Pat<(int_nvvm_floor_ftz_f Float32Regs:$a),
          (CVT_f32_f32 Float32Regs:$a, CvtRMI_FTZ)>;
def : Pat<(int_nvvm_floor_f Float32Regs:$a),
          (CVT_f32_f32 Float32Regs:$a, CvtRMI)>;
def : Pat<(int_nvvm_floor_d Float64Regs:$a),
          (CVT_f64_f64 Float64Regs:$a, CvtRMI)>;

def : Pat<(int_nvvm_ceil_ftz_f Float32Regs:$a),
          (CVT_f32_f32 Float32Regs:$a, CvtRPI_FTZ)>;
def : Pat<(int_nvvm_ceil_f Float32Regs:$a),
          (CVT_f32_f32 Float32Regs:$a, CvtRPI)>;
def : Pat<(int_nvvm_ceil_d Float64Regs:$a),
          (CVT_f64_f64 Float64Regs:$a, CvtRPI)>;

//
// Abs
//

def INT_NVVM_ABS_I : F_MATH_1<"abs.s32 \t$dst, $src0;", Int32Regs, Int32Regs,
  int_nvvm_abs_i>;
def INT_NVVM_ABS_LL : F_MATH_1<"abs.s64 \t$dst, $src0;", Int64Regs, Int64Regs,
  int_nvvm_abs_ll>;

def INT_NVVM_FABS_FTZ_F : F_MATH_1<"abs.ftz.f32 \t$dst, $src0;", Float32Regs,
  Float32Regs, int_nvvm_fabs_ftz_f>;
def INT_NVVM_FABS_F : F_MATH_1<"abs.f32 \t$dst, $src0;", Float32Regs,
  Float32Regs, int_nvvm_fabs_f>;

def INT_NVVM_FABS_D : F_MATH_1<"abs.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_fabs_d>;

//
// Round
//

def : Pat<(int_nvvm_round_ftz_f Float32Regs:$a),
          (CVT_f32_f32 Float32Regs:$a, CvtRNI_FTZ)>;
def : Pat<(int_nvvm_round_f Float32Regs:$a),
          (CVT_f32_f32 Float32Regs:$a, CvtRNI)>;
def : Pat<(int_nvvm_round_d Float64Regs:$a),
          (CVT_f64_f64 Float64Regs:$a, CvtRNI)>;

//
// Trunc
//

def : Pat<(int_nvvm_trunc_ftz_f Float32Regs:$a),
          (CVT_f32_f32 Float32Regs:$a, CvtRZI_FTZ)>;
def : Pat<(int_nvvm_trunc_f Float32Regs:$a),
          (CVT_f32_f32 Float32Regs:$a, CvtRZI)>;
def : Pat<(int_nvvm_trunc_d Float64Regs:$a),
          (CVT_f64_f64 Float64Regs:$a, CvtRZI)>;

//
// Saturate
//

def : Pat<(int_nvvm_saturate_ftz_f Float32Regs:$a),
          (CVT_f32_f32 Float32Regs:$a, CvtSAT_FTZ)>;
def : Pat<(int_nvvm_saturate_f Float32Regs:$a),
          (CVT_f32_f32 Float32Regs:$a, CvtSAT)>;
def : Pat<(int_nvvm_saturate_d Float64Regs:$a),
          (CVT_f64_f64 Float64Regs:$a, CvtSAT)>;

//
// Exp2  Log2
//

def INT_NVVM_EX2_APPROX_FTZ_F : F_MATH_1<"ex2.approx.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_ex2_approx_ftz_f>;
def INT_NVVM_EX2_APPROX_F : F_MATH_1<"ex2.approx.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_ex2_approx_f>;
def INT_NVVM_EX2_APPROX_D : F_MATH_1<"ex2.approx.f64 \t$dst, $src0;",
  Float64Regs, Float64Regs, int_nvvm_ex2_approx_d>;

def INT_NVVM_LG2_APPROX_FTZ_F : F_MATH_1<"lg2.approx.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_lg2_approx_ftz_f>;
def INT_NVVM_LG2_APPROX_F : F_MATH_1<"lg2.approx.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_lg2_approx_f>;
def INT_NVVM_LG2_APPROX_D : F_MATH_1<"lg2.approx.f64 \t$dst, $src0;",
  Float64Regs, Float64Regs, int_nvvm_lg2_approx_d>;

//
// Sin  Cos
//

def INT_NVVM_SIN_APPROX_FTZ_F : F_MATH_1<"sin.approx.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_sin_approx_ftz_f>;
def INT_NVVM_SIN_APPROX_F : F_MATH_1<"sin.approx.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_sin_approx_f>;

def INT_NVVM_COS_APPROX_FTZ_F : F_MATH_1<"cos.approx.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_cos_approx_ftz_f>;
def INT_NVVM_COS_APPROX_F : F_MATH_1<"cos.approx.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_cos_approx_f>;

//
// Fma
//

def INT_NVVM_FMA_RN_FTZ_F
  : F_MATH_3<"fma.rn.ftz.f32 \t$dst, $src0, $src1, $src2;", Float32Regs,
    Float32Regs, Float32Regs, Float32Regs, int_nvvm_fma_rn_ftz_f>;
def INT_NVVM_FMA_RN_F : F_MATH_3<"fma.rn.f32 \t$dst, $src0, $src1, $src2;",
  Float32Regs, Float32Regs, Float32Regs, Float32Regs, int_nvvm_fma_rn_f>;
def INT_NVVM_FMA_RZ_FTZ_F
  : F_MATH_3<"fma.rz.ftz.f32 \t$dst, $src0, $src1, $src2;", Float32Regs,
    Float32Regs, Float32Regs, Float32Regs, int_nvvm_fma_rz_ftz_f>;
def INT_NVVM_FMA_RZ_F : F_MATH_3<"fma.rz.f32 \t$dst, $src0, $src1, $src2;",
  Float32Regs, Float32Regs, Float32Regs, Float32Regs, int_nvvm_fma_rz_f>;
def INT_NVVM_FMA_RM_FTZ_F
  : F_MATH_3<"fma.rm.ftz.f32 \t$dst, $src0, $src1, $src2;", Float32Regs,
    Float32Regs, Float32Regs, Float32Regs, int_nvvm_fma_rm_ftz_f>;
def INT_NVVM_FMA_RM_F : F_MATH_3<"fma.rm.f32 \t$dst, $src0, $src1, $src2;",
  Float32Regs, Float32Regs, Float32Regs, Float32Regs, int_nvvm_fma_rm_f>;
def INT_NVVM_FMA_RP_FTZ_F
  : F_MATH_3<"fma.rp.ftz.f32 \t$dst, $src0, $src1, $src2;", Float32Regs,
    Float32Regs, Float32Regs, Float32Regs, int_nvvm_fma_rp_ftz_f>;
def INT_NVVM_FMA_RP_F : F_MATH_3<"fma.rp.f32 \t$dst, $src0, $src1, $src2;",
  Float32Regs, Float32Regs, Float32Regs, Float32Regs, int_nvvm_fma_rp_f>;

def INT_NVVM_FMA_RN_D : F_MATH_3<"fma.rn.f64 \t$dst, $src0, $src1, $src2;",
  Float64Regs, Float64Regs, Float64Regs, Float64Regs, int_nvvm_fma_rn_d>;
def INT_NVVM_FMA_RZ_D : F_MATH_3<"fma.rz.f64 \t$dst, $src0, $src1, $src2;",
  Float64Regs, Float64Regs, Float64Regs, Float64Regs, int_nvvm_fma_rz_d>;
def INT_NVVM_FMA_RM_D : F_MATH_3<"fma.rm.f64 \t$dst, $src0, $src1, $src2;",
  Float64Regs, Float64Regs, Float64Regs, Float64Regs, int_nvvm_fma_rm_d>;
def INT_NVVM_FMA_RP_D : F_MATH_3<"fma.rp.f64 \t$dst, $src0, $src1, $src2;",
  Float64Regs, Float64Regs, Float64Regs, Float64Regs, int_nvvm_fma_rp_d>;

//
// Rcp
//

def INT_NVVM_RCP_RN_FTZ_F : F_MATH_1<"rcp.rn.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rcp_rn_ftz_f>;
def INT_NVVM_RCP_RN_F : F_MATH_1<"rcp.rn.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rcp_rn_f>;
def INT_NVVM_RCP_RZ_FTZ_F : F_MATH_1<"rcp.rz.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rcp_rz_ftz_f>;
def INT_NVVM_RCP_RZ_F : F_MATH_1<"rcp.rz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rcp_rz_f>;
def INT_NVVM_RCP_RM_FTZ_F : F_MATH_1<"rcp.rm.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rcp_rm_ftz_f>;
def INT_NVVM_RCP_RM_F : F_MATH_1<"rcp.rm.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rcp_rm_f>;
def INT_NVVM_RCP_RP_FTZ_F : F_MATH_1<"rcp.rp.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rcp_rp_ftz_f>;
def INT_NVVM_RCP_RP_F : F_MATH_1<"rcp.rp.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rcp_rp_f>;

def INT_NVVM_RCP_RN_D : F_MATH_1<"rcp.rn.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_rcp_rn_d>;
def INT_NVVM_RCP_RZ_D : F_MATH_1<"rcp.rz.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_rcp_rz_d>;
def INT_NVVM_RCP_RM_D : F_MATH_1<"rcp.rm.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_rcp_rm_d>;
def INT_NVVM_RCP_RP_D : F_MATH_1<"rcp.rp.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_rcp_rp_d>;

def INT_NVVM_RCP_APPROX_FTZ_D : F_MATH_1<"rcp.approx.ftz.f64 \t$dst, $src0;",
  Float64Regs, Float64Regs, int_nvvm_rcp_approx_ftz_d>;

//
// Sqrt
//

def INT_NVVM_SQRT_RN_FTZ_F : F_MATH_1<"sqrt.rn.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_sqrt_rn_ftz_f>;
def INT_NVVM_SQRT_RN_F : F_MATH_1<"sqrt.rn.f32 \t$dst, $src0;", Float32Regs,
  Float32Regs, int_nvvm_sqrt_rn_f>;
def INT_NVVM_SQRT_RZ_FTZ_F : F_MATH_1<"sqrt.rz.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_sqrt_rz_ftz_f>;
def INT_NVVM_SQRT_RZ_F : F_MATH_1<"sqrt.rz.f32 \t$dst, $src0;", Float32Regs,
  Float32Regs, int_nvvm_sqrt_rz_f>;
def INT_NVVM_SQRT_RM_FTZ_F : F_MATH_1<"sqrt.rm.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_sqrt_rm_ftz_f>;
def INT_NVVM_SQRT_RM_F : F_MATH_1<"sqrt.rm.f32 \t$dst, $src0;", Float32Regs,
  Float32Regs, int_nvvm_sqrt_rm_f>;
def INT_NVVM_SQRT_RP_FTZ_F : F_MATH_1<"sqrt.rp.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_sqrt_rp_ftz_f>;
def INT_NVVM_SQRT_RP_F : F_MATH_1<"sqrt.rp.f32 \t$dst, $src0;", Float32Regs,
  Float32Regs, int_nvvm_sqrt_rp_f>;
def INT_NVVM_SQRT_APPROX_FTZ_F : F_MATH_1<"sqrt.approx.ftz.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_sqrt_approx_ftz_f>;
def INT_NVVM_SQRT_APPROX_F : F_MATH_1<"sqrt.approx.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_sqrt_approx_f>;

def INT_NVVM_SQRT_RN_D : F_MATH_1<"sqrt.rn.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_sqrt_rn_d>;
def INT_NVVM_SQRT_RZ_D : F_MATH_1<"sqrt.rz.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_sqrt_rz_d>;
def INT_NVVM_SQRT_RM_D : F_MATH_1<"sqrt.rm.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_sqrt_rm_d>;
def INT_NVVM_SQRT_RP_D : F_MATH_1<"sqrt.rp.f64 \t$dst, $src0;", Float64Regs,
  Float64Regs, int_nvvm_sqrt_rp_d>;

// nvvm_sqrt intrinsic
def : Pat<(int_nvvm_sqrt_f Float32Regs:$a),
          (INT_NVVM_SQRT_RN_FTZ_F Float32Regs:$a)>, Requires<[doF32FTZ, do_SQRTF32_RN]>;
def : Pat<(int_nvvm_sqrt_f Float32Regs:$a),
          (INT_NVVM_SQRT_RN_F Float32Regs:$a)>, Requires<[do_SQRTF32_RN]>;
def : Pat<(int_nvvm_sqrt_f Float32Regs:$a),
          (INT_NVVM_SQRT_APPROX_FTZ_F Float32Regs:$a)>, Requires<[doF32FTZ]>;
def : Pat<(int_nvvm_sqrt_f Float32Regs:$a),
          (INT_NVVM_SQRT_APPROX_F Float32Regs:$a)>;

//
// Rsqrt
//

def INT_NVVM_RSQRT_APPROX_FTZ_F
  : F_MATH_1<"rsqrt.approx.ftz.f32 \t$dst, $src0;", Float32Regs, Float32Regs,
    int_nvvm_rsqrt_approx_ftz_f>;
def INT_NVVM_RSQRT_APPROX_F : F_MATH_1<"rsqrt.approx.f32 \t$dst, $src0;",
  Float32Regs, Float32Regs, int_nvvm_rsqrt_approx_f>;
def INT_NVVM_RSQRT_APPROX_D : F_MATH_1<"rsqrt.approx.f64 \t$dst, $src0;",
  Float64Regs, Float64Regs, int_nvvm_rsqrt_approx_d>;

//
// Add
//

def INT_NVVM_ADD_RN_FTZ_F : F_MATH_2<"add.rn.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_add_rn_ftz_f>;
def INT_NVVM_ADD_RN_F : F_MATH_2<"add.rn.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_add_rn_f>;
def INT_NVVM_ADD_RZ_FTZ_F : F_MATH_2<"add.rz.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_add_rz_ftz_f>;
def INT_NVVM_ADD_RZ_F : F_MATH_2<"add.rz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_add_rz_f>;
def INT_NVVM_ADD_RM_FTZ_F : F_MATH_2<"add.rm.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_add_rm_ftz_f>;
def INT_NVVM_ADD_RM_F : F_MATH_2<"add.rm.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_add_rm_f>;
def INT_NVVM_ADD_RP_FTZ_F : F_MATH_2<"add.rp.ftz.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_add_rp_ftz_f>;
def INT_NVVM_ADD_RP_F : F_MATH_2<"add.rp.f32 \t$dst, $src0, $src1;",
  Float32Regs, Float32Regs, Float32Regs, int_nvvm_add_rp_f>;

def INT_NVVM_ADD_RN_D : F_MATH_2<"add.rn.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_add_rn_d>;
def INT_NVVM_ADD_RZ_D : F_MATH_2<"add.rz.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_add_rz_d>;
def INT_NVVM_ADD_RM_D : F_MATH_2<"add.rm.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_add_rm_d>;
def INT_NVVM_ADD_RP_D : F_MATH_2<"add.rp.f64 \t$dst, $src0, $src1;",
  Float64Regs, Float64Regs, Float64Regs, int_nvvm_add_rp_d>;

//
// Convert
//

def : Pat<(int_nvvm_d2f_rn_ftz Float64Regs:$a),
          (CVT_f32_f64 Float64Regs:$a, CvtRN_FTZ)>;
def : Pat<(int_nvvm_d2f_rn Float64Regs:$a),
          (CVT_f32_f64 Float64Regs:$a, CvtRN)>;
def : Pat<(int_nvvm_d2f_rz_ftz Float64Regs:$a),
          (CVT_f32_f64 Float64Regs:$a, CvtRZ_FTZ)>;
def : Pat<(int_nvvm_d2f_rz Float64Regs:$a),
          (CVT_f32_f64 Float64Regs:$a, CvtRZ)>;
def : Pat<(int_nvvm_d2f_rm_ftz Float64Regs:$a),
          (CVT_f32_f64 Float64Regs:$a, CvtRM_FTZ)>;
def : Pat<(int_nvvm_d2f_rm Float64Regs:$a),
          (CVT_f32_f64 Float64Regs:$a, CvtRM)>;
def : Pat<(int_nvvm_d2f_rp_ftz Float64Regs:$a),
          (CVT_f32_f64 Float64Regs:$a, CvtRP_FTZ)>;
def : Pat<(int_nvvm_d2f_rp Float64Regs:$a),
          (CVT_f32_f64 Float64Regs:$a, CvtRP)>;

def : Pat<(int_nvvm_d2i_rn Float64Regs:$a),
          (CVT_s32_f64 Float64Regs:$a, CvtRNI)>;
def : Pat<(int_nvvm_d2i_rz Float64Regs:$a),
          (CVT_s32_f64 Float64Regs:$a, CvtRZI)>;
def : Pat<(int_nvvm_d2i_rm Float64Regs:$a),
          (CVT_s32_f64 Float64Regs:$a, CvtRMI)>;
def : Pat<(int_nvvm_d2i_rp Float64Regs:$a),
          (CVT_s32_f64 Float64Regs:$a, CvtRPI)>;

def : Pat<(int_nvvm_d2ui_rn Float64Regs:$a),
          (CVT_u32_f64 Float64Regs:$a, CvtRNI)>;
def : Pat<(int_nvvm_d2ui_rz Float64Regs:$a),
          (CVT_u32_f64 Float64Regs:$a, CvtRZI)>;
def : Pat<(int_nvvm_d2ui_rm Float64Regs:$a),
          (CVT_u32_f64 Float64Regs:$a, CvtRMI)>;
def : Pat<(int_nvvm_d2ui_rp Float64Regs:$a),
          (CVT_u32_f64 Float64Regs:$a, CvtRPI)>;

def : Pat<(int_nvvm_i2d_rn Int32Regs:$a),
          (CVT_f64_s32 Int32Regs:$a, CvtRN)>;
def : Pat<(int_nvvm_i2d_rz Int32Regs:$a),
          (CVT_f64_s32 Int32Regs:$a, CvtRZ)>;
def : Pat<(int_nvvm_i2d_rm Int32Regs:$a),
          (CVT_f64_s32 Int32Regs:$a, CvtRM)>;
def : Pat<(int_nvvm_i2d_rp Int32Regs:$a),
          (CVT_f64_s32 Int32Regs:$a, CvtRP)>;

def : Pat<(int_nvvm_ui2d_rn Int32Regs:$a),
          (CVT_f64_u32 Int32Regs:$a, CvtRN)>;
def : Pat<(int_nvvm_ui2d_rz Int32Regs:$a),
          (CVT_f64_u32 Int32Regs:$a, CvtRZ)>;
def : Pat<(int_nvvm_ui2d_rm Int32Regs:$a),
          (CVT_f64_u32 Int32Regs:$a, CvtRM)>;
def : Pat<(int_nvvm_ui2d_rp Int32Regs:$a),
          (CVT_f64_u32 Int32Regs:$a, CvtRP)>;

def : Pat<(int_nvvm_f2i_rn_ftz Float32Regs:$a),
          (CVT_s32_f32 Float32Regs:$a, CvtRNI_FTZ)>;
def : Pat<(int_nvvm_f2i_rn Float32Regs:$a),
          (CVT_s32_f32 Float32Regs:$a, CvtRNI)>;
def : Pat<(int_nvvm_f2i_rz_ftz Float32Regs:$a),
          (CVT_s32_f32 Float32Regs:$a, CvtRZI_FTZ)>;
def : Pat<(int_nvvm_f2i_rz Float32Regs:$a),
          (CVT_s32_f32 Float32Regs:$a, CvtRZI)>;
def : Pat<(int_nvvm_f2i_rm_ftz Float32Regs:$a),
          (CVT_s32_f32 Float32Regs:$a, CvtRMI_FTZ)>;
def : Pat<(int_nvvm_f2i_rm Float32Regs:$a),
          (CVT_s32_f32 Float32Regs:$a, CvtRMI)>;
def : Pat<(int_nvvm_f2i_rp_ftz Float32Regs:$a),
          (CVT_s32_f32 Float32Regs:$a, CvtRPI_FTZ)>;
def : Pat<(int_nvvm_f2i_rp Float32Regs:$a),
          (CVT_s32_f32 Float32Regs:$a, CvtRPI)>;

def : Pat<(int_nvvm_f2ui_rn_ftz Float32Regs:$a),
          (CVT_u32_f32 Float32Regs:$a, CvtRNI_FTZ)>;
def : Pat<(int_nvvm_f2ui_rn Float32Regs:$a),
          (CVT_u32_f32 Float32Regs:$a, CvtRNI)>;
def : Pat<(int_nvvm_f2ui_rz_ftz Float32Regs:$a),
          (CVT_u32_f32 Float32Regs:$a, CvtRZI_FTZ)>;
def : Pat<(int_nvvm_f2ui_rz Float32Regs:$a),
          (CVT_u32_f32 Float32Regs:$a, CvtRZI)>;
def : Pat<(int_nvvm_f2ui_rm_ftz Float32Regs:$a),
          (CVT_u32_f32 Float32Regs:$a, CvtRMI_FTZ)>;
def : Pat<(int_nvvm_f2ui_rm Float32Regs:$a),
          (CVT_u32_f32 Float32Regs:$a, CvtRMI)>;
def : Pat<(int_nvvm_f2ui_rp_ftz Float32Regs:$a),
          (CVT_u32_f32 Float32Regs:$a, CvtRPI_FTZ)>;
def : Pat<(int_nvvm_f2ui_rp Float32Regs:$a),
          (CVT_u32_f32 Float32Regs:$a, CvtRPI)>;

def : Pat<(int_nvvm_i2f_rn Int32Regs:$a),
          (CVT_f32_s32 Int32Regs:$a, CvtRN)>;
def : Pat<(int_nvvm_i2f_rz Int32Regs:$a),
          (CVT_f32_s32 Int32Regs:$a, CvtRZ)>;
def : Pat<(int_nvvm_i2f_rm Int32Regs:$a),
          (CVT_f32_s32 Int32Regs:$a, CvtRM)>;
def : Pat<(int_nvvm_i2f_rp Int32Regs:$a),
          (CVT_f32_s32 Int32Regs:$a, CvtRP)>;

def : Pat<(int_nvvm_ui2f_rn Int32Regs:$a),
          (CVT_f32_u32 Int32Regs:$a, CvtRN)>;
def : Pat<(int_nvvm_ui2f_rz Int32Regs:$a),
          (CVT_f32_u32 Int32Regs:$a, CvtRZ)>;
def : Pat<(int_nvvm_ui2f_rm Int32Regs:$a),
          (CVT_f32_u32 Int32Regs:$a, CvtRM)>;
def : Pat<(int_nvvm_ui2f_rp Int32Regs:$a),
          (CVT_f32_u32 Int32Regs:$a, CvtRP)>;

def INT_NVVM_LOHI_I2D : F_MATH_2<"mov.b64 \t$dst, {{$src0, $src1}};",
  Float64Regs, Int32Regs, Int32Regs, int_nvvm_lohi_i2d>;

def INT_NVVM_D2I_LO : F_MATH_1<!strconcat("{{\n\t",
                       !strconcat(".reg .b32 %temp; \n\t",
             !strconcat("mov.b64 \t{$dst, %temp}, $src0;\n\t",
               "}}"))),
             Int32Regs, Float64Regs, int_nvvm_d2i_lo>;
def INT_NVVM_D2I_HI : F_MATH_1<!strconcat("{{\n\t",
                       !strconcat(".reg .b32 %temp; \n\t",
                         !strconcat("mov.b64 \t{%temp, $dst}, $src0;\n\t",
                           "}}"))),
             Int32Regs, Float64Regs, int_nvvm_d2i_hi>;

def : Pat<(int_nvvm_f2ll_rn_ftz Float32Regs:$a),
          (CVT_s64_f32 Float32Regs:$a, CvtRNI_FTZ)>;
def : Pat<(int_nvvm_f2ll_rn Float32Regs:$a),
          (CVT_s64_f32 Float32Regs:$a, CvtRNI)>;
def : Pat<(int_nvvm_f2ll_rz_ftz Float32Regs:$a),
          (CVT_s64_f32 Float32Regs:$a, CvtRZI_FTZ)>;
def : Pat<(int_nvvm_f2ll_rz Float32Regs:$a),
          (CVT_s64_f32 Float32Regs:$a, CvtRZI)>;
def : Pat<(int_nvvm_f2ll_rm_ftz Float32Regs:$a),
          (CVT_s64_f32 Float32Regs:$a, CvtRMI_FTZ)>;
def : Pat<(int_nvvm_f2ll_rm Float32Regs:$a),
          (CVT_s64_f32 Float32Regs:$a, CvtRMI)>;
def : Pat<(int_nvvm_f2ll_rp_ftz Float32Regs:$a),
          (CVT_s64_f32 Float32Regs:$a, CvtRPI_FTZ)>;
def : Pat<(int_nvvm_f2ll_rp Float32Regs:$a),
          (CVT_s64_f32 Float32Regs:$a, CvtRPI)>;

def : Pat<(int_nvvm_f2ull_rn_ftz Float32Regs:$a),
          (CVT_u64_f32 Float32Regs:$a, CvtRNI_FTZ)>;
def : Pat<(int_nvvm_f2ull_rn Float32Regs:$a),
          (CVT_u64_f32 Float32Regs:$a, CvtRNI)>;
def : Pat<(int_nvvm_f2ull_rz_ftz Float32Regs:$a),
          (CVT_u64_f32 Float32Regs:$a, CvtRZI_FTZ)>;
def : Pat<(int_nvvm_f2ull_rz Float32Regs:$a),
          (CVT_u64_f32 Float32Regs:$a, CvtRZI)>;
def : Pat<(int_nvvm_f2ull_rm_ftz Float32Regs:$a),
          (CVT_u64_f32 Float32Regs:$a, CvtRMI_FTZ)>;
def : Pat<(int_nvvm_f2ull_rm Float32Regs:$a),
          (CVT_u64_f32 Float32Regs:$a, CvtRMI)>;
def : Pat<(int_nvvm_f2ull_rp_ftz Float32Regs:$a),
          (CVT_u64_f32 Float32Regs:$a, CvtRPI_FTZ)>;
def : Pat<(int_nvvm_f2ull_rp Float32Regs:$a),
          (CVT_u64_f32 Float32Regs:$a, CvtRPI)>;

def : Pat<(int_nvvm_d2ll_rn Float64Regs:$a),
          (CVT_s64_f64 Float64Regs:$a, CvtRNI)>;
def : Pat<(int_nvvm_d2ll_rz Float64Regs:$a),
          (CVT_s64_f64 Float64Regs:$a, CvtRZI)>;
def : Pat<(int_nvvm_d2ll_rm Float64Regs:$a),
          (CVT_s64_f64 Float64Regs:$a, CvtRMI)>;
def : Pat<(int_nvvm_d2ll_rp Float64Regs:$a),
          (CVT_s64_f64 Float64Regs:$a, CvtRPI)>;

def : Pat<(int_nvvm_d2ull_rn Float64Regs:$a),
          (CVT_u64_f64 Float64Regs:$a, CvtRNI)>;
def : Pat<(int_nvvm_d2ull_rz Float64Regs:$a),
          (CVT_u64_f64 Float64Regs:$a, CvtRZI)>;
def : Pat<(int_nvvm_d2ull_rm Float64Regs:$a),
          (CVT_u64_f64 Float64Regs:$a, CvtRMI)>;
def : Pat<(int_nvvm_d2ull_rp Float64Regs:$a),
          (CVT_u64_f64 Float64Regs:$a, CvtRPI)>;

def : Pat<(int_nvvm_ll2f_rn Int64Regs:$a),
          (CVT_f32_s64 Int64Regs:$a, CvtRN)>;
def : Pat<(int_nvvm_ll2f_rz Int64Regs:$a),
          (CVT_f32_s64 Int64Regs:$a, CvtRZ)>;
def : Pat<(int_nvvm_ll2f_rm Int64Regs:$a),
          (CVT_f32_s64 Int64Regs:$a, CvtRM)>;
def : Pat<(int_nvvm_ll2f_rp Int64Regs:$a),
          (CVT_f32_s64 Int64Regs:$a, CvtRP)>;

def : Pat<(int_nvvm_ull2f_rn Int64Regs:$a),
          (CVT_f32_u64 Int64Regs:$a, CvtRN)>;
def : Pat<(int_nvvm_ull2f_rz Int64Regs:$a),
          (CVT_f32_u64 Int64Regs:$a, CvtRZ)>;
def : Pat<(int_nvvm_ull2f_rm Int64Regs:$a),
          (CVT_f32_u64 Int64Regs:$a, CvtRM)>;
def : Pat<(int_nvvm_ull2f_rp Int64Regs:$a),
          (CVT_f32_u64 Int64Regs:$a, CvtRP)>;

def : Pat<(int_nvvm_ll2d_rn Int64Regs:$a),
          (CVT_f64_s64 Int64Regs:$a, CvtRN)>;
def : Pat<(int_nvvm_ll2d_rz Int64Regs:$a),
          (CVT_f64_s64 Int64Regs:$a, CvtRZ)>;
def : Pat<(int_nvvm_ll2d_rm Int64Regs:$a),
          (CVT_f64_s64 Int64Regs:$a, CvtRM)>;
def : Pat<(int_nvvm_ll2d_rp Int64Regs:$a),
          (CVT_f64_s64 Int64Regs:$a, CvtRP)>;

def : Pat<(int_nvvm_ull2d_rn Int64Regs:$a),
          (CVT_f64_u64 Int64Regs:$a, CvtRN)>;
def : Pat<(int_nvvm_ull2d_rz Int64Regs:$a),
          (CVT_f64_u64 Int64Regs:$a, CvtRZ)>;
def : Pat<(int_nvvm_ull2d_rm Int64Regs:$a),
          (CVT_f64_u64 Int64Regs:$a, CvtRM)>;
def : Pat<(int_nvvm_ull2d_rp Int64Regs:$a),
          (CVT_f64_u64 Int64Regs:$a, CvtRP)>;


// FIXME: Ideally, we could use these patterns instead of the scope-creating
// patterns, but ptxas does not like these since .s16 is not compatible with
// .f16.  The solution is to use .bXX for all integer register types, but we
// are not there yet.
//def : Pat<(int_nvvm_f2h_rn_ftz Float32Regs:$a),
//          (CVT_f16_f32 Float32Regs:$a, CvtRN_FTZ)>;
//def : Pat<(int_nvvm_f2h_rn Float32Regs:$a),
//          (CVT_f16_f32 Float32Regs:$a, CvtRN)>;
//
//def : Pat<(int_nvvm_h2f Int16Regs:$a),
//          (CVT_f32_f16 Int16Regs:$a, CvtNONE)>;

def INT_NVVM_F2H_RN_FTZ : F_MATH_1<!strconcat("{{\n\t",
                                   !strconcat(".reg .b16 %temp;\n\t",
           !strconcat("cvt.rn.ftz.f16.f32 \t%temp, $src0;\n\t",
           !strconcat("mov.b16 \t$dst, %temp;\n",
             "}}")))),
                                   Int16Regs, Float32Regs, int_nvvm_f2h_rn_ftz>;
def INT_NVVM_F2H_RN : F_MATH_1<!strconcat("{{\n\t",
                                   !strconcat(".reg .b16 %temp;\n\t",
           !strconcat("cvt.rn.f16.f32 \t%temp, $src0;\n\t",
           !strconcat("mov.b16 \t$dst, %temp;\n",
             "}}")))),
           Int16Regs, Float32Regs, int_nvvm_f2h_rn>;

def INT_NVVM_H2F : F_MATH_1<!strconcat("{{\n\t",
                            !strconcat(".reg .b16 %temp;\n\t",
          !strconcat("mov.b16 \t%temp, $src0;\n\t",
          !strconcat("cvt.f32.f16 \t$dst, %temp;\n\t",
            "}}")))),
          Float32Regs, Int16Regs, int_nvvm_h2f>;

def : Pat<(f32 (f16_to_f32 Int16Regs:$a)),
          (CVT_f32_f16 Int16Regs:$a, CvtNONE)>;
def : Pat<(i16 (f32_to_f16 Float32Regs:$a)),
          (CVT_f16_f32 Float32Regs:$a, CvtRN_FTZ)>, Requires<[doF32FTZ]>;
def : Pat<(i16 (f32_to_f16 Float32Regs:$a)),
          (CVT_f16_f32 Float32Regs:$a, CvtRN)>;

//
// Bitcast
//

def INT_NVVM_BITCAST_F2I : F_MATH_1<"mov.b32 \t$dst, $src0;", Int32Regs,
  Float32Regs, int_nvvm_bitcast_f2i>;
def INT_NVVM_BITCAST_I2F : F_MATH_1<"mov.b32 \t$dst, $src0;", Float32Regs,
  Int32Regs, int_nvvm_bitcast_i2f>;

def INT_NVVM_BITCAST_LL2D : F_MATH_1<"mov.b64 \t$dst, $src0;", Float64Regs,
  Int64Regs, int_nvvm_bitcast_ll2d>;
def INT_NVVM_BITCAST_D2LL : F_MATH_1<"mov.b64 \t$dst, $src0;", Int64Regs,
  Float64Regs, int_nvvm_bitcast_d2ll>;

//-----------------------------------
// Atomic Functions
//-----------------------------------

class ATOMIC_GLOBAL_CHK <dag ops, dag frag>
 : PatFrag<ops, frag, [{
   return ChkMemSDNodeAddressSpace(N, llvm::ADDRESS_SPACE_GLOBAL);
}]>;
class ATOMIC_SHARED_CHK <dag ops, dag frag>
 : PatFrag<ops, frag, [{
   return ChkMemSDNodeAddressSpace(N, llvm::ADDRESS_SPACE_SHARED);
}]>;
class ATOMIC_GENERIC_CHK <dag ops, dag frag>
 : PatFrag<ops, frag, [{
   return ChkMemSDNodeAddressSpace(N, llvm::ADDRESS_SPACE_GENERIC);
}]>;

multiclass F_ATOMIC_2_imp<NVPTXRegClass ptrclass, NVPTXRegClass regclass,
  string SpaceStr, string TypeStr, string OpcStr, PatFrag IntOp,
  Operand IMMType, SDNode IMM, Predicate Pred> {
  def reg : NVPTXInst<(outs regclass:$dst), (ins ptrclass:$addr, regclass:$b),
               !strconcat("atom",
         !strconcat(SpaceStr,
         !strconcat(OpcStr,
         !strconcat(TypeStr,
         !strconcat(" \t$dst, [$addr], $b;", ""))))),
         [(set regclass:$dst, (IntOp ptrclass:$addr, regclass:$b))]>,
  Requires<[Pred]>;
  def imm : NVPTXInst<(outs regclass:$dst), (ins ptrclass:$addr, IMMType:$b),
               !strconcat("atom",
         !strconcat(SpaceStr,
         !strconcat(OpcStr,
         !strconcat(TypeStr,
         !strconcat(" \t$dst, [$addr], $b;", ""))))),
         [(set regclass:$dst, (IntOp ptrclass:$addr, IMM:$b))]>,
  Requires<[Pred]>;
}
multiclass F_ATOMIC_2<NVPTXRegClass regclass, string SpaceStr, string TypeStr,
  string OpcStr, PatFrag IntOp, Operand IMMType, SDNode IMM, Predicate Pred> {
  defm p32 : F_ATOMIC_2_imp<Int32Regs, regclass, SpaceStr, TypeStr, OpcStr,
    IntOp, IMMType, IMM, Pred>;
  defm p64 : F_ATOMIC_2_imp<Int64Regs, regclass, SpaceStr, TypeStr, OpcStr,
    IntOp, IMMType, IMM, Pred>;
}

// has 2 operands, neg the second one
multiclass F_ATOMIC_2_NEG_imp<NVPTXRegClass ptrclass, NVPTXRegClass regclass,
  string SpaceStr, string TypeStr, string OpcStr, PatFrag IntOp,
  Operand IMMType, Predicate Pred> {
  def reg : NVPTXInst<(outs regclass:$dst), (ins ptrclass:$addr, regclass:$b),
    !strconcat("{{ \n\t",
         !strconcat(".reg \t.s",
         !strconcat(TypeStr,
         !strconcat(" temp; \n\t",
         !strconcat("neg.s",
         !strconcat(TypeStr,
         !strconcat(" \ttemp, $b; \n\t",
               !strconcat("atom",
         !strconcat(SpaceStr,
         !strconcat(OpcStr,
         !strconcat(".u",
         !strconcat(TypeStr,
         !strconcat(" \t$dst, [$addr], temp; \n\t",
           !strconcat("}}", "")))))))))))))),
         [(set regclass:$dst, (IntOp ptrclass:$addr, regclass:$b))]>,
  Requires<[Pred]>;
}
multiclass F_ATOMIC_2_NEG<NVPTXRegClass regclass, string SpaceStr,
  string TypeStr, string OpcStr, PatFrag IntOp, Operand IMMType,
  Predicate Pred> {
 defm p32: F_ATOMIC_2_NEG_imp<Int32Regs, regclass, SpaceStr, TypeStr, OpcStr,
   IntOp, IMMType, Pred> ;
 defm p64: F_ATOMIC_2_NEG_imp<Int64Regs, regclass, SpaceStr, TypeStr, OpcStr,
   IntOp, IMMType, Pred> ;
}

// has 3 operands
multiclass F_ATOMIC_3_imp<NVPTXRegClass ptrclass, NVPTXRegClass regclass,
  string SpaceStr, string TypeStr, string OpcStr, PatFrag IntOp,
  Operand IMMType, Predicate Pred> {
  def reg : NVPTXInst<(outs regclass:$dst),
    (ins ptrclass:$addr, regclass:$b, regclass:$c),
               !strconcat("atom",
         !strconcat(SpaceStr,
         !strconcat(OpcStr,
         !strconcat(TypeStr,
         !strconcat(" \t$dst, [$addr], $b, $c;", ""))))),
         [(set regclass:$dst,
           (IntOp ptrclass:$addr, regclass:$b, regclass:$c))]>,
         Requires<[Pred]>;
  def imm1 : NVPTXInst<(outs regclass:$dst),
    (ins ptrclass:$addr, IMMType:$b, regclass:$c),
               !strconcat("atom",
         !strconcat(SpaceStr,
         !strconcat(OpcStr,
         !strconcat(TypeStr,
         !strconcat(" \t$dst, [$addr], $b, $c;", ""))))),
         [(set regclass:$dst, (IntOp ptrclass:$addr, imm:$b, regclass:$c))]>,
  Requires<[Pred]>;
  def imm2 : NVPTXInst<(outs regclass:$dst),
    (ins ptrclass:$addr, regclass:$b, IMMType:$c),
               !strconcat("atom",
         !strconcat(SpaceStr,
         !strconcat(OpcStr,
         !strconcat(TypeStr,
         !strconcat(" \t$dst, [$addr], $b, $c;", ""))))),
         [(set regclass:$dst, (IntOp ptrclass:$addr, regclass:$b, imm:$c))]>,
  Requires<[Pred]>;
  def imm3 : NVPTXInst<(outs regclass:$dst),
    (ins ptrclass:$addr, IMMType:$b, IMMType:$c),
               !strconcat("atom",
         !strconcat(SpaceStr,
         !strconcat(OpcStr,
         !strconcat(TypeStr,
         !strconcat(" \t$dst, [$addr], $b, $c;", ""))))),
         [(set regclass:$dst, (IntOp ptrclass:$addr, imm:$b, imm:$c))]>,
  Requires<[Pred]>;
}
multiclass F_ATOMIC_3<NVPTXRegClass regclass, string SpaceStr, string TypeStr,
  string OpcStr, PatFrag IntOp, Operand IMMType, Predicate Pred> {
  defm p32 : F_ATOMIC_3_imp<Int32Regs, regclass, SpaceStr, TypeStr, OpcStr,
    IntOp, IMMType, Pred>;
  defm p64 : F_ATOMIC_3_imp<Int64Regs, regclass, SpaceStr, TypeStr, OpcStr,
    IntOp, IMMType, Pred>;
}

// atom_add

def atomic_load_add_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_add_32 node:$a, node:$b)>;
def atomic_load_add_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_add_32 node:$a, node:$b)>;
def atomic_load_add_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_add_32 node:$a, node:$b)>;
def atomic_load_add_64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_add_64 node:$a, node:$b)>;
def atomic_load_add_64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_add_64 node:$a, node:$b)>;
def atomic_load_add_64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_add_64 node:$a, node:$b)>;
def atomic_load_add_f32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (int_nvvm_atomic_load_add_f32 node:$a, node:$b)>;
def atomic_load_add_f32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (int_nvvm_atomic_load_add_f32 node:$a, node:$b)>;
def atomic_load_add_f32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (int_nvvm_atomic_load_add_f32 node:$a, node:$b)>;

defm INT_PTX_ATOM_ADD_G_32 : F_ATOMIC_2<Int32Regs, ".global", ".u32", ".add",
  atomic_load_add_32_g, i32imm, imm, hasAtomRedG32>;
defm INT_PTX_ATOM_ADD_S_32 : F_ATOMIC_2<Int32Regs, ".shared", ".u32", ".add",
  atomic_load_add_32_s, i32imm, imm, hasAtomRedS32>;
defm INT_PTX_ATOM_ADD_GEN_32 : F_ATOMIC_2<Int32Regs, "", ".u32", ".add",
  atomic_load_add_32_gen, i32imm, imm, hasAtomRedGen32>;
defm INT_PTX_ATOM_ADD_GEN_32_USE_G : F_ATOMIC_2<Int32Regs, ".global", ".u32",
  ".add", atomic_load_add_32_gen, i32imm, imm, useAtomRedG32forGen32>;

defm INT_PTX_ATOM_ADD_G_64 : F_ATOMIC_2<Int64Regs, ".global", ".u64", ".add",
  atomic_load_add_64_g, i64imm, imm, hasAtomRedG64>;
defm INT_PTX_ATOM_ADD_S_64 : F_ATOMIC_2<Int64Regs, ".shared", ".u64", ".add",
  atomic_load_add_64_s, i64imm, imm, hasAtomRedS64>;
defm INT_PTX_ATOM_ADD_GEN_64 : F_ATOMIC_2<Int64Regs, "", ".u64", ".add",
  atomic_load_add_64_gen, i64imm, imm, hasAtomRedGen64>;
defm INT_PTX_ATOM_ADD_GEN_64_USE_G : F_ATOMIC_2<Int64Regs, ".global", ".u64",
  ".add", atomic_load_add_64_gen, i64imm, imm, useAtomRedG64forGen64>;

defm INT_PTX_ATOM_ADD_G_F32 : F_ATOMIC_2<Float32Regs, ".global", ".f32", ".add",
  atomic_load_add_f32_g, f32imm, fpimm, hasAtomAddF32>;
defm INT_PTX_ATOM_ADD_S_F32 : F_ATOMIC_2<Float32Regs, ".shared", ".f32", ".add",
  atomic_load_add_f32_s, f32imm, fpimm, hasAtomAddF32>;
defm INT_PTX_ATOM_ADD_GEN_F32 : F_ATOMIC_2<Float32Regs, "", ".f32", ".add",
  atomic_load_add_f32_gen, f32imm, fpimm, hasAtomAddF32>;

// atom_sub

def atomic_load_sub_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_32 node:$a, node:$b)>;
def atomic_load_sub_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_32 node:$a, node:$b)>;
def atomic_load_sub_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_32 node:$a, node:$b)>;
def atomic_load_sub_64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_64 node:$a, node:$b)>;
def atomic_load_sub_64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_64 node:$a, node:$b)>;
def atomic_load_sub_64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_sub_64 node:$a, node:$b)>;

defm INT_PTX_ATOM_SUB_G_32 : F_ATOMIC_2_NEG<Int32Regs, ".global", "32", ".add",
  atomic_load_sub_32_g, i32imm, hasAtomRedG32>;
defm INT_PTX_ATOM_SUB_G_64 : F_ATOMIC_2_NEG<Int64Regs, ".global", "64", ".add",
  atomic_load_sub_64_g, i64imm, hasAtomRedG64>;
defm INT_PTX_ATOM_SUB_GEN_32 : F_ATOMIC_2_NEG<Int32Regs, "", "32", ".add",
  atomic_load_sub_32_gen, i32imm, hasAtomRedGen32>;
defm INT_PTX_ATOM_SUB_GEN_32_USE_G : F_ATOMIC_2_NEG<Int32Regs, ".global", "32",
  ".add", atomic_load_sub_32_gen, i32imm, useAtomRedG32forGen32>;
defm INT_PTX_ATOM_SUB_S_32 : F_ATOMIC_2_NEG<Int32Regs, ".shared", "32", ".add",
  atomic_load_sub_32_s, i32imm, hasAtomRedS32>;
defm INT_PTX_ATOM_SUB_S_64 : F_ATOMIC_2_NEG<Int64Regs, ".shared", "64", ".add",
  atomic_load_sub_64_s, i64imm, hasAtomRedS64>;
defm INT_PTX_ATOM_SUB_GEN_64 : F_ATOMIC_2_NEG<Int64Regs, "", "64", ".add",
  atomic_load_sub_64_gen, i64imm, hasAtomRedGen64>;
defm INT_PTX_ATOM_SUB_GEN_64_USE_G : F_ATOMIC_2_NEG<Int64Regs, ".global", "64",
  ".add", atomic_load_sub_64_gen, i64imm, useAtomRedG64forGen64>;

// atom_swap

def atomic_swap_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_swap_32 node:$a, node:$b)>;
def atomic_swap_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_swap_32 node:$a, node:$b)>;
def atomic_swap_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_swap_32 node:$a, node:$b)>;
def atomic_swap_64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_swap_64 node:$a, node:$b)>;
def atomic_swap_64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_swap_64 node:$a, node:$b)>;
def atomic_swap_64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_swap_64 node:$a, node:$b)>;

defm INT_PTX_ATOM_SWAP_G_32 : F_ATOMIC_2<Int32Regs, ".global", ".b32", ".exch",
  atomic_swap_32_g, i32imm, imm, hasAtomRedG32>;
defm INT_PTX_ATOM_SWAP_S_32 : F_ATOMIC_2<Int32Regs, ".shared", ".b32", ".exch",
  atomic_swap_32_s, i32imm, imm, hasAtomRedS32>;
defm INT_PTX_ATOM_SWAP_GEN_32 : F_ATOMIC_2<Int32Regs, "", ".b32", ".exch",
  atomic_swap_32_gen, i32imm, imm, hasAtomRedGen32>;
defm INT_PTX_ATOM_SWAP_GEN_32_USE_G : F_ATOMIC_2<Int32Regs, ".global", ".b32",
  ".exch", atomic_swap_32_gen, i32imm, imm, useAtomRedG32forGen32>;
defm INT_PTX_ATOM_SWAP_G_64 : F_ATOMIC_2<Int64Regs, ".global", ".b64", ".exch",
  atomic_swap_64_g, i64imm, imm, hasAtomRedG64>;
defm INT_PTX_ATOM_SWAP_S_64 : F_ATOMIC_2<Int64Regs, ".shared", ".b64", ".exch",
  atomic_swap_64_s, i64imm, imm, hasAtomRedS64>;
defm INT_PTX_ATOM_SWAP_GEN_64 : F_ATOMIC_2<Int64Regs, "", ".b64", ".exch",
  atomic_swap_64_gen, i64imm, imm, hasAtomRedGen64>;
defm INT_PTX_ATOM_SWAP_GEN_64_USE_G : F_ATOMIC_2<Int64Regs, ".global", ".b64",
  ".exch", atomic_swap_64_gen, i64imm, imm, useAtomRedG64forGen64>;

// atom_max

def atomic_load_max_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b)
  , (atomic_load_max_32 node:$a, node:$b)>;
def atomic_load_max_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_max_32 node:$a, node:$b)>;
def atomic_load_max_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_max_32 node:$a, node:$b)>;
def atomic_load_umax_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_umax_32 node:$a, node:$b)>;
def atomic_load_umax_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_umax_32 node:$a, node:$b)>;
def atomic_load_umax_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_umax_32 node:$a, node:$b)>;

defm INT_PTX_ATOM_LOAD_MAX_G_32 : F_ATOMIC_2<Int32Regs, ".global", ".s32",
  ".max", atomic_load_max_32_g, i32imm, imm, hasAtomRedG32>;
defm INT_PTX_ATOM_LOAD_MAX_S_32 : F_ATOMIC_2<Int32Regs, ".shared", ".s32",
  ".max", atomic_load_max_32_s, i32imm, imm, hasAtomRedS32>;
defm INT_PTX_ATOM_LOAD_MAX_GEN_32 : F_ATOMIC_2<Int32Regs, "", ".s32", ".max",
  atomic_load_max_32_gen, i32imm, imm, hasAtomRedGen32>;
defm INT_PTX_ATOM_LOAD_MAX_GEN_32_USE_G : F_ATOMIC_2<Int32Regs, ".global",
  ".s32", ".max", atomic_load_max_32_gen, i32imm, imm, useAtomRedG32forGen32>;
defm INT_PTX_ATOM_LOAD_UMAX_G_32 : F_ATOMIC_2<Int32Regs, ".global", ".u32",
  ".max", atomic_load_umax_32_g, i32imm, imm, hasAtomRedG32>;
defm INT_PTX_ATOM_LOAD_UMAX_S_32 : F_ATOMIC_2<Int32Regs, ".shared", ".u32",
  ".max", atomic_load_umax_32_s, i32imm, imm, hasAtomRedS32>;
defm INT_PTX_ATOM_LOAD_UMAX_GEN_32 : F_ATOMIC_2<Int32Regs, "", ".u32", ".max",
  atomic_load_umax_32_gen, i32imm, imm, hasAtomRedGen32>;
defm INT_PTX_ATOM_LOAD_UMAX_GEN_32_USE_G : F_ATOMIC_2<Int32Regs, ".global",
  ".u32", ".max", atomic_load_umax_32_gen, i32imm, imm, useAtomRedG32forGen32>;

// atom_min

def atomic_load_min_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_min_32 node:$a, node:$b)>;
def atomic_load_min_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_min_32 node:$a, node:$b)>;
def atomic_load_min_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_min_32 node:$a, node:$b)>;
def atomic_load_umin_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_umin_32 node:$a, node:$b)>;
def atomic_load_umin_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_umin_32 node:$a, node:$b)>;
def atomic_load_umin_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_umin_32 node:$a, node:$b)>;

defm INT_PTX_ATOM_LOAD_MIN_G_32 : F_ATOMIC_2<Int32Regs, ".global", ".s32",
  ".min", atomic_load_min_32_g, i32imm, imm, hasAtomRedG32>;
defm INT_PTX_ATOM_LOAD_MIN_S_32 : F_ATOMIC_2<Int32Regs, ".shared", ".s32",
  ".min", atomic_load_min_32_s, i32imm, imm, hasAtomRedS32>;
defm INT_PTX_ATOM_LOAD_MIN_GEN_32 : F_ATOMIC_2<Int32Regs, "", ".s32", ".min",
  atomic_load_min_32_gen, i32imm, imm, hasAtomRedGen32>;
defm INT_PTX_ATOM_LOAD_MIN_GEN_32_USE_G : F_ATOMIC_2<Int32Regs, ".global",
  ".s32", ".min", atomic_load_min_32_gen, i32imm, imm, useAtomRedG32forGen32>;
defm INT_PTX_ATOM_LOAD_UMIN_G_32 : F_ATOMIC_2<Int32Regs, ".global", ".u32",
  ".min", atomic_load_umin_32_g, i32imm, imm, hasAtomRedG32>;
defm INT_PTX_ATOM_LOAD_UMIN_S_32 : F_ATOMIC_2<Int32Regs, ".shared", ".u32",
  ".min", atomic_load_umin_32_s, i32imm, imm, hasAtomRedS32>;
defm INT_PTX_ATOM_LOAD_UMIN_GEN_32 : F_ATOMIC_2<Int32Regs, "", ".u32", ".min",
  atomic_load_umin_32_gen, i32imm, imm, hasAtomRedGen32>;
defm INT_PTX_ATOM_LOAD_UMIN_GEN_32_USE_G : F_ATOMIC_2<Int32Regs, ".global",
  ".u32", ".min", atomic_load_umin_32_gen, i32imm, imm, useAtomRedG32forGen32>;

// atom_inc  atom_dec

def atomic_load_inc_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (int_nvvm_atomic_load_inc_32 node:$a, node:$b)>;
def atomic_load_inc_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (int_nvvm_atomic_load_inc_32 node:$a, node:$b)>;
def atomic_load_inc_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (int_nvvm_atomic_load_inc_32 node:$a, node:$b)>;
def atomic_load_dec_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (int_nvvm_atomic_load_dec_32 node:$a, node:$b)>;
def atomic_load_dec_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (int_nvvm_atomic_load_dec_32 node:$a, node:$b)>;
def atomic_load_dec_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (int_nvvm_atomic_load_dec_32 node:$a, node:$b)>;

defm INT_PTX_ATOM_INC_G_32 : F_ATOMIC_2<Int32Regs, ".global", ".u32", ".inc",
  atomic_load_inc_32_g, i32imm, imm, hasAtomRedG32>;
defm INT_PTX_ATOM_INC_S_32 : F_ATOMIC_2<Int32Regs, ".shared", ".u32", ".inc",
  atomic_load_inc_32_s, i32imm, imm, hasAtomRedS32>;
defm INT_PTX_ATOM_INC_GEN_32 : F_ATOMIC_2<Int32Regs, "", ".u32", ".inc",
  atomic_load_inc_32_gen, i32imm, imm, hasAtomRedGen32>;
defm INT_PTX_ATOM_INC_GEN_32_USE_G : F_ATOMIC_2<Int32Regs, ".global", ".u32",
  ".inc", atomic_load_inc_32_gen, i32imm, imm, useAtomRedG32forGen32>;
defm INT_PTX_ATOM_DEC_G_32 : F_ATOMIC_2<Int32Regs, ".global", ".u32", ".dec",
  atomic_load_dec_32_g, i32imm, imm, hasAtomRedG32>;
defm INT_PTX_ATOM_DEC_S_32 : F_ATOMIC_2<Int32Regs, ".shared", ".u32", ".dec",
  atomic_load_dec_32_s, i32imm, imm, hasAtomRedS32>;
defm INT_PTX_ATOM_DEC_GEN_32 : F_ATOMIC_2<Int32Regs, "", ".u32", ".dec",
  atomic_load_dec_32_gen, i32imm, imm, hasAtomRedGen32>;
defm INT_PTX_ATOM_DEC_GEN_32_USE_G : F_ATOMIC_2<Int32Regs, ".global", ".u32",
  ".dec", atomic_load_dec_32_gen, i32imm, imm, useAtomRedG32forGen32>;

// atom_and

def atomic_load_and_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_and_32 node:$a, node:$b)>;
def atomic_load_and_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_and_32 node:$a, node:$b)>;
def atomic_load_and_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_and_32 node:$a, node:$b)>;

defm INT_PTX_ATOM_AND_G_32 : F_ATOMIC_2<Int32Regs, ".global", ".b32", ".and",
  atomic_load_and_32_g, i32imm, imm, hasAtomRedG32>;
defm INT_PTX_ATOM_AND_S_32 : F_ATOMIC_2<Int32Regs, ".shared", ".b32", ".and",
  atomic_load_and_32_s, i32imm, imm, hasAtomRedS32>;
defm INT_PTX_ATOM_AND_GEN_32 : F_ATOMIC_2<Int32Regs, "", ".b32", ".and",
  atomic_load_and_32_gen, i32imm, imm, hasAtomRedGen32>;
defm INT_PTX_ATOM_AND_GEN_32_USE_G : F_ATOMIC_2<Int32Regs, ".global", ".b32",
  ".and", atomic_load_and_32_gen, i32imm, imm, useAtomRedG32forGen32>;

// atom_or

def atomic_load_or_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_or_32 node:$a, node:$b)>;
def atomic_load_or_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_or_32 node:$a, node:$b)>;
def atomic_load_or_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_or_32 node:$a, node:$b)>;

defm INT_PTX_ATOM_OR_G_32 : F_ATOMIC_2<Int32Regs, ".global", ".b32", ".or",
  atomic_load_or_32_g, i32imm, imm, hasAtomRedG32>;
defm INT_PTX_ATOM_OR_GEN_32 : F_ATOMIC_2<Int32Regs, "", ".b32", ".or",
  atomic_load_or_32_gen, i32imm, imm, hasAtomRedGen32>;
defm INT_PTX_ATOM_OR_GEN_32_USE_G : F_ATOMIC_2<Int32Regs, ".global", ".b32",
  ".or", atomic_load_or_32_gen, i32imm, imm, useAtomRedG32forGen32>;
defm INT_PTX_ATOM_OR_S_32 : F_ATOMIC_2<Int32Regs, ".shared", ".b32", ".or",
  atomic_load_or_32_s, i32imm, imm, hasAtomRedS32>;

// atom_xor

def atomic_load_xor_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b),
  (atomic_load_xor_32 node:$a, node:$b)>;
def atomic_load_xor_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b),
  (atomic_load_xor_32 node:$a, node:$b)>;
def atomic_load_xor_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b),
  (atomic_load_xor_32 node:$a, node:$b)>;

defm INT_PTX_ATOM_XOR_G_32 : F_ATOMIC_2<Int32Regs, ".global", ".b32", ".xor",
  atomic_load_xor_32_g, i32imm, imm, hasAtomRedG32>;
defm INT_PTX_ATOM_XOR_S_32 : F_ATOMIC_2<Int32Regs, ".shared", ".b32", ".xor",
  atomic_load_xor_32_s, i32imm, imm, hasAtomRedS32>;
defm INT_PTX_ATOM_XOR_GEN_32 : F_ATOMIC_2<Int32Regs, "", ".b32", ".xor",
  atomic_load_xor_32_gen, i32imm, imm, hasAtomRedGen32>;
defm INT_PTX_ATOM_XOR_GEN_32_USE_G : F_ATOMIC_2<Int32Regs, ".global", ".b32",
  ".xor", atomic_load_xor_32_gen, i32imm, imm, useAtomRedG32forGen32>;

// atom_cas

def atomic_cmp_swap_32_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_32 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_32_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_32 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_32_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_32 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_64_g: ATOMIC_GLOBAL_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_64 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_64_s: ATOMIC_SHARED_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_64 node:$a, node:$b, node:$c)>;
def atomic_cmp_swap_64_gen: ATOMIC_GENERIC_CHK<(ops node:$a, node:$b, node:$c),
  (atomic_cmp_swap_64 node:$a, node:$b, node:$c)>;

defm INT_PTX_ATOM_CAS_G_32 : F_ATOMIC_3<Int32Regs, ".global", ".b32", ".cas",
  atomic_cmp_swap_32_g, i32imm, hasAtomRedG32>;
defm INT_PTX_ATOM_CAS_S_32 : F_ATOMIC_3<Int32Regs, ".shared", ".b32", ".cas",
  atomic_cmp_swap_32_s, i32imm, hasAtomRedS32>;
defm INT_PTX_ATOM_CAS_GEN_32 : F_ATOMIC_3<Int32Regs, "", ".b32", ".cas",
  atomic_cmp_swap_32_gen, i32imm, hasAtomRedGen32>;
defm INT_PTX_ATOM_CAS_GEN_32_USE_G : F_ATOMIC_3<Int32Regs, ".global", ".b32",
  ".cas", atomic_cmp_swap_32_gen, i32imm, useAtomRedG32forGen32>;
defm INT_PTX_ATOM_CAS_G_64 : F_ATOMIC_3<Int64Regs, ".global", ".b64", ".cas",
  atomic_cmp_swap_64_g, i64imm, hasAtomRedG64>;
defm INT_PTX_ATOM_CAS_S_64 : F_ATOMIC_3<Int64Regs, ".shared", ".b64", ".cas",
  atomic_cmp_swap_64_s, i64imm, hasAtomRedS64>;
defm INT_PTX_ATOM_CAS_GEN_64 : F_ATOMIC_3<Int64Regs, "", ".b64", ".cas",
  atomic_cmp_swap_64_gen, i64imm, hasAtomRedGen64>;
defm INT_PTX_ATOM_CAS_GEN_64_USE_G : F_ATOMIC_3<Int64Regs, ".global", ".b64",
  ".cas", atomic_cmp_swap_64_gen, i64imm, useAtomRedG64forGen64>;


//-----------------------------------
// Read Special Registers
//-----------------------------------
class F_SREG<string OpStr, NVPTXRegClass regclassOut, Intrinsic IntOp> :
      NVPTXInst<(outs regclassOut:$dst), (ins),
               OpStr,
         [(set regclassOut:$dst, (IntOp))]>;

def INT_PTX_SREG_TID_X : F_SREG<"mov.u32 \t$dst, %tid.x;", Int32Regs,
  int_nvvm_read_ptx_sreg_tid_x>;
def INT_PTX_SREG_TID_Y : F_SREG<"mov.u32 \t$dst, %tid.y;", Int32Regs,
  int_nvvm_read_ptx_sreg_tid_y>;
def INT_PTX_SREG_TID_Z : F_SREG<"mov.u32 \t$dst, %tid.z;", Int32Regs,
  int_nvvm_read_ptx_sreg_tid_z>;

def INT_PTX_SREG_NTID_X : F_SREG<"mov.u32 \t$dst, %ntid.x;", Int32Regs,
  int_nvvm_read_ptx_sreg_ntid_x>;
def INT_PTX_SREG_NTID_Y : F_SREG<"mov.u32 \t$dst, %ntid.y;", Int32Regs,
  int_nvvm_read_ptx_sreg_ntid_y>;
def INT_PTX_SREG_NTID_Z : F_SREG<"mov.u32 \t$dst, %ntid.z;", Int32Regs,
  int_nvvm_read_ptx_sreg_ntid_z>;

def INT_PTX_SREG_CTAID_X : F_SREG<"mov.u32 \t$dst, %ctaid.x;", Int32Regs,
  int_nvvm_read_ptx_sreg_ctaid_x>;
def INT_PTX_SREG_CTAID_Y : F_SREG<"mov.u32 \t$dst, %ctaid.y;", Int32Regs,
  int_nvvm_read_ptx_sreg_ctaid_y>;
def INT_PTX_SREG_CTAID_Z : F_SREG<"mov.u32 \t$dst, %ctaid.z;", Int32Regs,
  int_nvvm_read_ptx_sreg_ctaid_z>;

def INT_PTX_SREG_NCTAID_X : F_SREG<"mov.u32 \t$dst, %nctaid.x;", Int32Regs,
  int_nvvm_read_ptx_sreg_nctaid_x>;
def INT_PTX_SREG_NCTAID_Y : F_SREG<"mov.u32 \t$dst, %nctaid.y;", Int32Regs,
  int_nvvm_read_ptx_sreg_nctaid_y>;
def INT_PTX_SREG_NCTAID_Z : F_SREG<"mov.u32 \t$dst, %nctaid.z;", Int32Regs,
  int_nvvm_read_ptx_sreg_nctaid_z>;

def INT_PTX_SREG_WARPSIZE : F_SREG<"mov.u32 \t$dst, WARP_SZ;", Int32Regs,
  int_nvvm_read_ptx_sreg_warpsize>;


//-----------------------------------
// Support for ldu on sm_20 or later
//-----------------------------------

def ldu_i8 : PatFrag<(ops node:$ptr), (int_nvvm_ldu_global_i node:$ptr), [{
  MemIntrinsicSDNode *M = cast<MemIntrinsicSDNode>(N);
  return M->getMemoryVT() == MVT::i8;
}]>;

// Scalar
// @TODO: Revisit this, Changed imemAny to imem
multiclass LDU_G<string TyStr, NVPTXRegClass regclass, Intrinsic IntOp> {
  def areg: NVPTXInst<(outs regclass:$result), (ins Int32Regs:$src),
               !strconcat("ldu.global.", TyStr),
         [(set regclass:$result, (IntOp Int32Regs:$src))]>, Requires<[hasLDU]>;
  def areg64: NVPTXInst<(outs regclass:$result), (ins Int64Regs:$src),
               !strconcat("ldu.global.", TyStr),
         [(set regclass:$result, (IntOp Int64Regs:$src))]>, Requires<[hasLDU]>;
 def avar:  NVPTXInst<(outs regclass:$result), (ins imem:$src),
               !strconcat("ldu.global.", TyStr),
                [(set regclass:$result, (IntOp (Wrapper tglobaladdr:$src)))]>,
                Requires<[hasLDU]>;
 def ari :  NVPTXInst<(outs regclass:$result), (ins MEMri:$src),
               !strconcat("ldu.global.", TyStr),
         [(set regclass:$result, (IntOp ADDRri:$src))]>, Requires<[hasLDU]>;
 def ari64 :  NVPTXInst<(outs regclass:$result), (ins MEMri64:$src),
               !strconcat("ldu.global.", TyStr),
         [(set regclass:$result, (IntOp ADDRri64:$src))]>, Requires<[hasLDU]>;
}

multiclass LDU_G_NOINTRIN<string TyStr, NVPTXRegClass regclass, PatFrag IntOp> {
  def areg: NVPTXInst<(outs regclass:$result), (ins Int32Regs:$src),
               !strconcat("ldu.global.", TyStr),
         [(set regclass:$result, (IntOp Int32Regs:$src))]>, Requires<[hasLDU]>;
  def areg64: NVPTXInst<(outs regclass:$result), (ins Int64Regs:$src),
               !strconcat("ldu.global.", TyStr),
         [(set regclass:$result, (IntOp Int64Regs:$src))]>, Requires<[hasLDU]>;
 def avar:  NVPTXInst<(outs regclass:$result), (ins imem:$src),
               !strconcat("ldu.global.", TyStr),
         [(set regclass:$result, (IntOp (Wrapper tglobaladdr:$src)))]>,
         Requires<[hasLDU]>;
 def ari :  NVPTXInst<(outs regclass:$result), (ins MEMri:$src),
               !strconcat("ldu.global.", TyStr),
         [(set regclass:$result, (IntOp ADDRri:$src))]>, Requires<[hasLDU]>;
 def ari64 :  NVPTXInst<(outs regclass:$result), (ins MEMri64:$src),
               !strconcat("ldu.global.", TyStr),
         [(set regclass:$result, (IntOp ADDRri64:$src))]>, Requires<[hasLDU]>;
}

defm INT_PTX_LDU_GLOBAL_i8  : LDU_G_NOINTRIN<"u8 \t$result, [$src];", Int16Regs,
                                             ldu_i8>;
defm INT_PTX_LDU_GLOBAL_i16 : LDU_G<"u16 \t$result, [$src];", Int16Regs,
int_nvvm_ldu_global_i>;
defm INT_PTX_LDU_GLOBAL_i32 : LDU_G<"u32 \t$result, [$src];", Int32Regs,
int_nvvm_ldu_global_i>;
defm INT_PTX_LDU_GLOBAL_i64 : LDU_G<"u64 \t$result, [$src];", Int64Regs,
int_nvvm_ldu_global_i>;
defm INT_PTX_LDU_GLOBAL_f32 : LDU_G<"f32 \t$result, [$src];", Float32Regs,
int_nvvm_ldu_global_f>;
defm INT_PTX_LDU_GLOBAL_f64 : LDU_G<"f64 \t$result, [$src];", Float64Regs,
int_nvvm_ldu_global_f>;
defm INT_PTX_LDU_GLOBAL_p32 : LDU_G<"u32 \t$result, [$src];", Int32Regs,
int_nvvm_ldu_global_p>;
defm INT_PTX_LDU_GLOBAL_p64 : LDU_G<"u64 \t$result, [$src];", Int64Regs,
int_nvvm_ldu_global_p>;

// vector

// Elementized vector ldu
multiclass VLDU_G_ELE_V2<string TyStr, NVPTXRegClass regclass> {
 def _areg32: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins Int32Regs:$src),
                     !strconcat("ldu.global.", TyStr), []>;
 def _areg64: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins Int64Regs:$src),
                     !strconcat("ldu.global.", TyStr), []>;
 def _ari32: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins MEMri:$src),
                     !strconcat("ldu.global.", TyStr), []>;
 def _ari64: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins MEMri64:$src),
                     !strconcat("ldu.global.", TyStr), []>;
 def _avar: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins imemAny:$src),
                     !strconcat("ldu.global.", TyStr), []>;
}

multiclass VLDU_G_ELE_V4<string TyStr, NVPTXRegClass regclass> { 
 def _areg32: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                            regclass:$dst4), (ins Int32Regs:$src), 
               !strconcat("ldu.global.", TyStr), []>;
 def _areg64: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                            regclass:$dst4), (ins Int64Regs:$src), 
               !strconcat("ldu.global.", TyStr), []>;
 def _ari32: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                            regclass:$dst4), (ins MEMri:$src), 
               !strconcat("ldu.global.", TyStr), []>;
 def _ari64: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                            regclass:$dst4), (ins MEMri64:$src), 
               !strconcat("ldu.global.", TyStr), []>;
 def _avar: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                            regclass:$dst4), (ins imemAny:$src), 
               !strconcat("ldu.global.", TyStr), []>;
}

defm INT_PTX_LDU_G_v2i8_ELE
  : VLDU_G_ELE_V2<"v2.u8 \t{{$dst1, $dst2}}, [$src];",  Int16Regs>;
defm INT_PTX_LDU_G_v2i16_ELE
  : VLDU_G_ELE_V2<"v2.u16 \t{{$dst1, $dst2}}, [$src];", Int16Regs>;
defm INT_PTX_LDU_G_v2i32_ELE
  : VLDU_G_ELE_V2<"v2.u32 \t{{$dst1, $dst2}}, [$src];", Int32Regs>;
defm INT_PTX_LDU_G_v2f32_ELE
  : VLDU_G_ELE_V2<"v2.f32 \t{{$dst1, $dst2}}, [$src];", Float32Regs>;
defm INT_PTX_LDU_G_v2i64_ELE
  : VLDU_G_ELE_V2<"v2.u64 \t{{$dst1, $dst2}}, [$src];", Int64Regs>;
defm INT_PTX_LDU_G_v2f64_ELE
  : VLDU_G_ELE_V2<"v2.f64 \t{{$dst1, $dst2}}, [$src];", Float64Regs>;
defm INT_PTX_LDU_G_v4i8_ELE
  : VLDU_G_ELE_V4<"v4.u8 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];", Int16Regs>;
defm INT_PTX_LDU_G_v4i16_ELE
  : VLDU_G_ELE_V4<"v4.u16 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];",
    Int16Regs>;
defm INT_PTX_LDU_G_v4i32_ELE
  : VLDU_G_ELE_V4<"v4.u32 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];",
    Int32Regs>;
defm INT_PTX_LDU_G_v4f32_ELE
  : VLDU_G_ELE_V4<"v4.f32 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];",
    Float32Regs>;


//-----------------------------------
// Support for ldg on sm_35 or later 
//-----------------------------------

def ldg_i8 : PatFrag<(ops node:$ptr), (int_nvvm_ldg_global_i node:$ptr), [{
  MemIntrinsicSDNode *M = cast<MemIntrinsicSDNode>(N);
  return M->getMemoryVT() == MVT::i8;
}]>;

multiclass LDG_G<string TyStr, NVPTXRegClass regclass, Intrinsic IntOp> {
  def areg: NVPTXInst<(outs regclass:$result), (ins Int32Regs:$src),
               !strconcat("ld.global.nc.", TyStr),
         [(set regclass:$result, (IntOp Int32Regs:$src))]>, Requires<[hasLDG]>;
  def areg64: NVPTXInst<(outs regclass:$result), (ins Int64Regs:$src),
               !strconcat("ld.global.nc.", TyStr),
         [(set regclass:$result, (IntOp Int64Regs:$src))]>, Requires<[hasLDG]>;
 def avar:  NVPTXInst<(outs regclass:$result), (ins imem:$src),
               !strconcat("ld.global.nc.", TyStr),
         [(set regclass:$result, (IntOp (Wrapper tglobaladdr:$src)))]>,
         Requires<[hasLDG]>;
 def ari :  NVPTXInst<(outs regclass:$result), (ins MEMri:$src),
               !strconcat("ld.global.nc.", TyStr),
         [(set regclass:$result, (IntOp ADDRri:$src))]>, Requires<[hasLDG]>;
 def ari64 :  NVPTXInst<(outs regclass:$result), (ins MEMri64:$src),
               !strconcat("ld.global.nc.", TyStr),
         [(set regclass:$result, (IntOp ADDRri64:$src))]>, Requires<[hasLDG]>;
}

multiclass LDG_G_NOINTRIN<string TyStr, NVPTXRegClass regclass, PatFrag IntOp> {
  def areg: NVPTXInst<(outs regclass:$result), (ins Int32Regs:$src),
               !strconcat("ld.global.nc.", TyStr),
         [(set regclass:$result, (IntOp Int32Regs:$src))]>, Requires<[hasLDG]>;
  def areg64: NVPTXInst<(outs regclass:$result), (ins Int64Regs:$src),
               !strconcat("ld.global.nc.", TyStr),
         [(set regclass:$result, (IntOp Int64Regs:$src))]>, Requires<[hasLDG]>;
 def avar:  NVPTXInst<(outs regclass:$result), (ins imem:$src),
               !strconcat("ld.global.nc.", TyStr),
         [(set regclass:$result, (IntOp (Wrapper tglobaladdr:$src)))]>,
        Requires<[hasLDG]>;
 def ari :  NVPTXInst<(outs regclass:$result), (ins MEMri:$src),
               !strconcat("ld.global.nc.", TyStr),
         [(set regclass:$result, (IntOp ADDRri:$src))]>, Requires<[hasLDG]>;
 def ari64 :  NVPTXInst<(outs regclass:$result), (ins MEMri64:$src),
               !strconcat("ld.global.nc.", TyStr),
         [(set regclass:$result, (IntOp ADDRri64:$src))]>, Requires<[hasLDG]>;
}

defm INT_PTX_LDG_GLOBAL_i8
  : LDG_G_NOINTRIN<"u8 \t$result, [$src];",  Int16Regs, ldg_i8>;
defm INT_PTX_LDG_GLOBAL_i16
  : LDG_G<"u16 \t$result, [$src];", Int16Regs,   int_nvvm_ldg_global_i>;
defm INT_PTX_LDG_GLOBAL_i32
  : LDG_G<"u32 \t$result, [$src];", Int32Regs,   int_nvvm_ldg_global_i>;
defm INT_PTX_LDG_GLOBAL_i64
  : LDG_G<"u64 \t$result, [$src];", Int64Regs,   int_nvvm_ldg_global_i>;
defm INT_PTX_LDG_GLOBAL_f32
  : LDG_G<"f32 \t$result, [$src];", Float32Regs, int_nvvm_ldg_global_f>;
defm INT_PTX_LDG_GLOBAL_f64
  : LDG_G<"f64 \t$result, [$src];", Float64Regs, int_nvvm_ldg_global_f>;
defm INT_PTX_LDG_GLOBAL_p32
  : LDG_G<"u32 \t$result, [$src];", Int32Regs,   int_nvvm_ldg_global_p>;
defm INT_PTX_LDG_GLOBAL_p64
  : LDG_G<"u64 \t$result, [$src];", Int64Regs,   int_nvvm_ldg_global_p>;

// vector

// Elementized vector ldg 
multiclass VLDG_G_ELE_V2<string TyStr, NVPTXRegClass regclass> {
 def _areg32: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins Int32Regs:$src),
                     !strconcat("ld.global.nc.", TyStr), []>;
 def _areg64: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins Int64Regs:$src),
                     !strconcat("ld.global.nc.", TyStr), []>;
 def _ari32: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins MEMri:$src),
                     !strconcat("ld.global.nc.", TyStr), []>;
 def _ari64: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins MEMri64:$src),
                     !strconcat("ld.global.nc.", TyStr), []>;
 def _avar: NVPTXInst<(outs regclass:$dst1, regclass:$dst2),
                     (ins imemAny:$src),
                     !strconcat("ld.global.nc.", TyStr), []>;
}

multiclass VLDG_G_ELE_V4<string TyStr, NVPTXRegClass regclass> { 
  def _areg32: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                              regclass:$dst4), (ins Int32Regs:$src), 
               !strconcat("ld.global.nc.", TyStr), []>;
  def _areg64: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                               regclass:$dst4), (ins Int64Regs:$src), 
               !strconcat("ld.global.nc.", TyStr), []>;
  def _ari32: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                              regclass:$dst4), (ins MEMri:$src), 
               !strconcat("ld.global.nc.", TyStr), []>;
  def _ari64: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                              regclass:$dst4), (ins MEMri64:$src), 
               !strconcat("ld.global.nc.", TyStr), []>;
  def _avar: NVPTXInst<(outs regclass:$dst1, regclass:$dst2, regclass:$dst3,
                             regclass:$dst4), (ins imemAny:$src), 
               !strconcat("ld.global.nc.", TyStr), []>;
}

// FIXME: 8-bit LDG should be fixed once LDG/LDU nodes are made into proper loads.
defm INT_PTX_LDG_G_v2i8_ELE
  : VLDG_G_ELE_V2<"v2.u8 \t{{$dst1, $dst2}}, [$src];",  Int16Regs>;
defm INT_PTX_LDG_G_v2i16_ELE
  : VLDG_G_ELE_V2<"v2.u16 \t{{$dst1, $dst2}}, [$src];", Int16Regs>;
defm INT_PTX_LDG_G_v2i32_ELE
  : VLDG_G_ELE_V2<"v2.u32 \t{{$dst1, $dst2}}, [$src];", Int32Regs>;
defm INT_PTX_LDG_G_v2f32_ELE
  : VLDG_G_ELE_V2<"v2.f32 \t{{$dst1, $dst2}}, [$src];", Float32Regs>;
defm INT_PTX_LDG_G_v2i64_ELE
  : VLDG_G_ELE_V2<"v2.u64 \t{{$dst1, $dst2}}, [$src];", Int64Regs>;
defm INT_PTX_LDG_G_v2f64_ELE
  : VLDG_G_ELE_V2<"v2.f64 \t{{$dst1, $dst2}}, [$src];", Float64Regs>;
defm INT_PTX_LDG_G_v4i8_ELE
  : VLDG_G_ELE_V4<"v4.u8 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];", Int16Regs>;
defm INT_PTX_LDG_G_v4i16_ELE
  : VLDG_G_ELE_V4<"v4.u16 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];", Int16Regs>;
defm INT_PTX_LDG_G_v4i32_ELE
  : VLDG_G_ELE_V4<"v4.u32 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];", Int32Regs>;
defm INT_PTX_LDG_G_v4f32_ELE
  : VLDG_G_ELE_V4<"v4.f32 \t{{$dst1, $dst2, $dst3, $dst4}}, [$src];", Float32Regs>;


multiclass NG_TO_G<string Str, Intrinsic Intrin> {
   def _yes : NVPTXInst<(outs Int32Regs:$result), (ins Int32Regs:$src),
          !strconcat("cvta.", !strconcat(Str, ".u32 \t$result, $src;")),
      [(set Int32Regs:$result, (Intrin Int32Regs:$src))]>,
   Requires<[hasGenericLdSt]>;
   def _yes_64 : NVPTXInst<(outs Int64Regs:$result), (ins Int64Regs:$src),
          !strconcat("cvta.", !strconcat(Str, ".u64 \t$result, $src;")),
      [(set Int64Regs:$result, (Intrin Int64Regs:$src))]>,
   Requires<[hasGenericLdSt]>;

// @TODO: Are these actually needed?  I believe global addresses will be copied
// to register values anyway.
   /*def __addr_yes : NVPTXInst<(outs Int32Regs:$result), (ins imemAny:$src),
          !strconcat("cvta.", !strconcat(Str, ".u32 \t$result, $src;")),
      [(set Int32Regs:$result, (Intrin (Wrapper tglobaladdr:$src)))]>,
      Requires<[hasGenericLdSt]>;
   def __addr_yes_64 : NVPTXInst<(outs Int64Regs:$result), (ins imemAny:$src),
          !strconcat("cvta.", !strconcat(Str, ".u64 \t$result, $src;")),
      [(set Int64Regs:$result, (Intrin (Wrapper tglobaladdr:$src)))]>,
      Requires<[hasGenericLdSt]>;*/

   def _no : NVPTXInst<(outs Int32Regs:$result), (ins Int32Regs:$src),
          "mov.u32 \t$result, $src;",
      [(set Int32Regs:$result, (Intrin Int32Regs:$src))]>;
   def _no_64 : NVPTXInst<(outs Int64Regs:$result), (ins Int64Regs:$src),
          "mov.u64 \t$result, $src;",
      [(set Int64Regs:$result, (Intrin Int64Regs:$src))]>;

// @TODO: Are these actually needed?  I believe global addresses will be copied
// to register values anyway.
   /*def _addr_no : NVPTXInst<(outs Int32Regs:$result), (ins imem:$src),
          "mov.u32 \t$result, $src;",
      [(set Int32Regs:$result, (Intrin (Wrapper tglobaladdr:$src)))]>;
   def _addr_no_64 : NVPTXInst<(outs Int64Regs:$result), (ins imem:$src),
          "mov.u64 \t$result, $src;",
      [(set Int64Regs:$result, (Intrin (Wrapper tglobaladdr:$src)))]>;*/
}

multiclass G_TO_NG<string Str, Intrinsic Intrin> {
   def _yes : NVPTXInst<(outs Int32Regs:$result), (ins Int32Regs:$src),
          !strconcat("cvta.to.", !strconcat(Str, ".u32 \t$result, $src;")),
      [(set Int32Regs:$result, (Intrin Int32Regs:$src))]>,
   Requires<[hasGenericLdSt]>;
   def _yes_64 : NVPTXInst<(outs Int64Regs:$result), (ins Int64Regs:$src),
          !strconcat("cvta.to.", !strconcat(Str, ".u64 \t$result, $src;")),
      [(set Int64Regs:$result, (Intrin Int64Regs:$src))]>,
   Requires<[hasGenericLdSt]>;
   def _no : NVPTXInst<(outs Int32Regs:$result), (ins Int32Regs:$src),
          "mov.u32 \t$result, $src;",
      [(set Int32Regs:$result, (Intrin Int32Regs:$src))]>;
   def _no_64 : NVPTXInst<(outs Int64Regs:$result), (ins Int64Regs:$src),
          "mov.u64 \t$result, $src;",
      [(set Int64Regs:$result, (Intrin Int64Regs:$src))]>;
}

defm cvta_local  : NG_TO_G<"local", int_nvvm_ptr_local_to_gen>;
defm cvta_shared : NG_TO_G<"shared", int_nvvm_ptr_shared_to_gen>;
defm cvta_global : NG_TO_G<"global", int_nvvm_ptr_global_to_gen>;
defm cvta_const  : NG_TO_G<"const", int_nvvm_ptr_constant_to_gen>;

defm cvta_to_local   : G_TO_NG<"local", int_nvvm_ptr_gen_to_local>;
defm cvta_to_shared : G_TO_NG<"shared", int_nvvm_ptr_gen_to_shared>;
defm cvta_to_global : G_TO_NG<"global", int_nvvm_ptr_gen_to_global>;
defm cvta_to_const  : G_TO_NG<"const", int_nvvm_ptr_gen_to_constant>;


// nvvm.ptr.gen.to.param
def nvvm_ptr_gen_to_param : NVPTXInst<(outs Int32Regs:$result),
  (ins Int32Regs:$src),
                        "mov.u32 \t$result, $src;",
                              [(set Int32Regs:$result,
                                (int_nvvm_ptr_gen_to_param Int32Regs:$src))]>;
def nvvm_ptr_gen_to_param_64 : NVPTXInst<(outs Int64Regs:$result),
  (ins Int64Regs:$src),
                        "mov.u64 \t$result, $src;",
                              [(set Int64Regs:$result,
                                (int_nvvm_ptr_gen_to_param Int64Regs:$src))]>;


// nvvm.move intrinsicc
def nvvm_move_i16 : NVPTXInst<(outs Int16Regs:$r), (ins Int16Regs:$s),
                             "mov.b16 \t$r, $s;",
                             [(set Int16Regs:$r,
                               (int_nvvm_move_i16 Int16Regs:$s))]>;
def nvvm_move_i32 : NVPTXInst<(outs Int32Regs:$r), (ins Int32Regs:$s),
                             "mov.b32 \t$r, $s;",
                             [(set Int32Regs:$r,
                               (int_nvvm_move_i32 Int32Regs:$s))]>;
def nvvm_move_i64 : NVPTXInst<(outs Int64Regs:$r), (ins Int64Regs:$s),
                             "mov.b64 \t$r, $s;",
                             [(set Int64Regs:$r,
                               (int_nvvm_move_i64 Int64Regs:$s))]>;
def nvvm_move_float : NVPTXInst<(outs Float32Regs:$r), (ins Float32Regs:$s),
                             "mov.f32 \t$r, $s;",
                             [(set Float32Regs:$r,
                               (int_nvvm_move_float Float32Regs:$s))]>;
def nvvm_move_double : NVPTXInst<(outs Float64Regs:$r), (ins Float64Regs:$s),
                             "mov.f64 \t$r, $s;",
                             [(set Float64Regs:$r,
                               (int_nvvm_move_double Float64Regs:$s))]>;
def nvvm_move_ptr32 : NVPTXInst<(outs Int32Regs:$r), (ins Int32Regs:$s),
                             "mov.u32 \t$r, $s;",
                             [(set Int32Regs:$r,
                               (int_nvvm_move_ptr Int32Regs:$s))]>;
def nvvm_move_ptr64 : NVPTXInst<(outs Int64Regs:$r), (ins Int64Regs:$s),
                             "mov.u64 \t$r, $s;",
                             [(set Int64Regs:$r,
                               (int_nvvm_move_ptr Int64Regs:$s))]>;

// @TODO: Are these actually needed, or will we always just see symbols
// copied to registers first?
/*def nvvm_move_sym32 : NVPTXInst<(outs Int32Regs:$r), (ins imem:$s),
                             "mov.u32 \t$r, $s;",
                             [(set Int32Regs:$r,
                             (int_nvvm_move_ptr texternalsym:$s))]>;
def nvvm_move_sym64 : NVPTXInst<(outs Int64Regs:$r), (ins imem:$s),
                             "mov.u64 \t$r, $s;",
                             [(set Int64Regs:$r,
                             (int_nvvm_move_ptr texternalsym:$s))]>;*/


// MoveParam        %r1, param
// ptr_local_to_gen %r2, %r1
// ptr_gen_to_local %r3, %r2
// ->
// mov %r1, param

// @TODO: Revisit this.  There is a type
// contradiction between iPTRAny and iPTR for the addr defs, so the move_sym
// instructions are not currently defined. However, we can use the ptr
// variants and the asm printer will do the right thing.
def : Pat<(i64 (int_nvvm_ptr_gen_to_local (int_nvvm_ptr_local_to_gen
                (MoveParam texternalsym:$src)))),
               (nvvm_move_ptr64  texternalsym:$src)>;
def : Pat<(i32 (int_nvvm_ptr_gen_to_local (int_nvvm_ptr_local_to_gen
                (MoveParam texternalsym:$src)))),
               (nvvm_move_ptr32  texternalsym:$src)>;


//-----------------------------------
// Compiler Error Warn
// - Just ignore them in codegen
//-----------------------------------

def INT_NVVM_COMPILER_WARN_32 : NVPTXInst<(outs), (ins Int32Regs:$a),
                "// llvm.nvvm.compiler.warn()",
                [(int_nvvm_compiler_warn Int32Regs:$a)]>;
def INT_NVVM_COMPILER_WARN_64 : NVPTXInst<(outs), (ins Int64Regs:$a),
                "// llvm.nvvm.compiler.warn()",
                [(int_nvvm_compiler_warn Int64Regs:$a)]>;
def INT_NVVM_COMPILER_ERROR_32 : NVPTXInst<(outs), (ins Int32Regs:$a),
                "// llvm.nvvm.compiler.error()",
                [(int_nvvm_compiler_error Int32Regs:$a)]>;
def INT_NVVM_COMPILER_ERROR_64 : NVPTXInst<(outs), (ins Int64Regs:$a),
                "// llvm.nvvm.compiler.error()",
                [(int_nvvm_compiler_error Int64Regs:$a)]>;



//===-- Old PTX Back-end Intrinsics ---------------------------------------===//

// These intrinsics are handled to retain compatibility with the old backend.

// PTX Special Purpose Register Accessor Intrinsics

class PTX_READ_SPECIAL_REGISTER_R64<string regname, Intrinsic intop>
  : NVPTXInst<(outs Int64Regs:$d), (ins),
              !strconcat(!strconcat("mov.u64\t$d, %", regname), ";"),
              [(set Int64Regs:$d, (intop))]>;

class PTX_READ_SPECIAL_REGISTER_R32<string regname, Intrinsic intop>
  : NVPTXInst<(outs Int32Regs:$d), (ins),
              !strconcat(!strconcat("mov.u32\t$d, %", regname), ";"),
              [(set Int32Regs:$d, (intop))]>;

// TODO Add read vector-version of special registers

def PTX_READ_TID_X   : PTX_READ_SPECIAL_REGISTER_R32<"tid.x",
                                                     int_ptx_read_tid_x>;
def PTX_READ_TID_Y   : PTX_READ_SPECIAL_REGISTER_R32<"tid.y",
                                                     int_ptx_read_tid_y>;
def PTX_READ_TID_Z   : PTX_READ_SPECIAL_REGISTER_R32<"tid.z",
                                                     int_ptx_read_tid_z>;
def PTX_READ_TID_W   : PTX_READ_SPECIAL_REGISTER_R32<"tid.w",
                                                     int_ptx_read_tid_w>;

def PTX_READ_NTID_X   : PTX_READ_SPECIAL_REGISTER_R32<"ntid.x",
                                                      int_ptx_read_ntid_x>;
def PTX_READ_NTID_Y   : PTX_READ_SPECIAL_REGISTER_R32<"ntid.y",
                                                      int_ptx_read_ntid_y>;
def PTX_READ_NTID_Z   : PTX_READ_SPECIAL_REGISTER_R32<"ntid.z",
                                                      int_ptx_read_ntid_z>;
def PTX_READ_NTID_W   : PTX_READ_SPECIAL_REGISTER_R32<"ntid.w",
                                                      int_ptx_read_ntid_w>;

def PTX_READ_LANEID  : PTX_READ_SPECIAL_REGISTER_R32<"laneid",
                                                     int_ptx_read_laneid>;
def PTX_READ_WARPID  : PTX_READ_SPECIAL_REGISTER_R32<"warpid",
                                                     int_ptx_read_warpid>;
def PTX_READ_NWARPID : PTX_READ_SPECIAL_REGISTER_R32<"nwarpid",
                                                     int_ptx_read_nwarpid>;

def PTX_READ_CTAID_X   : PTX_READ_SPECIAL_REGISTER_R32<"ctaid.x",
                                                       int_ptx_read_ctaid_x>;
def PTX_READ_CTAID_Y   : PTX_READ_SPECIAL_REGISTER_R32<"ctaid.y",
                                                       int_ptx_read_ctaid_y>;
def PTX_READ_CTAID_Z   : PTX_READ_SPECIAL_REGISTER_R32<"ctaid.z",
                                                       int_ptx_read_ctaid_z>;
def PTX_READ_CTAID_W   : PTX_READ_SPECIAL_REGISTER_R32<"ctaid.w",
                                                       int_ptx_read_ctaid_w>;

def PTX_READ_NCTAID_X   : PTX_READ_SPECIAL_REGISTER_R32<"nctaid.x",
                                                        int_ptx_read_nctaid_x>;
def PTX_READ_NCTAID_Y   : PTX_READ_SPECIAL_REGISTER_R32<"nctaid.y",
                                                        int_ptx_read_nctaid_y>;
def PTX_READ_NCTAID_Z   : PTX_READ_SPECIAL_REGISTER_R32<"nctaid.z",
                                                        int_ptx_read_nctaid_z>;
def PTX_READ_NCTAID_W   : PTX_READ_SPECIAL_REGISTER_R32<"nctaid.w",
                                                        int_ptx_read_nctaid_w>;

def PTX_READ_SMID  : PTX_READ_SPECIAL_REGISTER_R32<"smid",
                                                   int_ptx_read_smid>;
def PTX_READ_NSMID  : PTX_READ_SPECIAL_REGISTER_R32<"nsmid",
                                                    int_ptx_read_nsmid>;
def PTX_READ_GRIDID  : PTX_READ_SPECIAL_REGISTER_R32<"gridid",
                                                     int_ptx_read_gridid>;

def PTX_READ_LANEMASK_EQ
  : PTX_READ_SPECIAL_REGISTER_R32<"lanemask_eq", int_ptx_read_lanemask_eq>;
def PTX_READ_LANEMASK_LE
  : PTX_READ_SPECIAL_REGISTER_R32<"lanemask_le", int_ptx_read_lanemask_le>;
def PTX_READ_LANEMASK_LT
  : PTX_READ_SPECIAL_REGISTER_R32<"lanemask_lt", int_ptx_read_lanemask_lt>;
def PTX_READ_LANEMASK_GE
  : PTX_READ_SPECIAL_REGISTER_R32<"lanemask_ge", int_ptx_read_lanemask_ge>;
def PTX_READ_LANEMASK_GT
  : PTX_READ_SPECIAL_REGISTER_R32<"lanemask_gt", int_ptx_read_lanemask_gt>;

def PTX_READ_CLOCK
  : PTX_READ_SPECIAL_REGISTER_R32<"clock", int_ptx_read_clock>;
def PTX_READ_CLOCK64
  : PTX_READ_SPECIAL_REGISTER_R64<"clock64", int_ptx_read_clock64>;

def PTX_READ_PM0 : PTX_READ_SPECIAL_REGISTER_R32<"pm0", int_ptx_read_pm0>;
def PTX_READ_PM1 : PTX_READ_SPECIAL_REGISTER_R32<"pm1", int_ptx_read_pm1>;
def PTX_READ_PM2 : PTX_READ_SPECIAL_REGISTER_R32<"pm2", int_ptx_read_pm2>;
def PTX_READ_PM3 : PTX_READ_SPECIAL_REGISTER_R32<"pm3", int_ptx_read_pm3>;

// PTX Parallel Synchronization and Communication Intrinsics

def PTX_BAR_SYNC : NVPTXInst<(outs), (ins i32imm:$i), "bar.sync\t$i;",
                             [(int_ptx_bar_sync imm:$i)]>;
