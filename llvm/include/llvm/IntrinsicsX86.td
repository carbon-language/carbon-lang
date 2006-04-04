//===- IntrinsicsX86.td - Defines X86 intrinsics -----------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines all of the X86-specific intrinsics.
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// SSE1

// Arithmetic ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_add_ss : GCCBuiltin<"__builtin_ia32_addss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_sub_ss : GCCBuiltin<"__builtin_ia32_subss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_mul_ss : GCCBuiltin<"__builtin_ia32_mulss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_div_ss : GCCBuiltin<"__builtin_ia32_divss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_sqrt_ss : GCCBuiltin<"__builtin_ia32_sqrtss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty],
                        [InstrNoMem]>;
  def int_x86_sse_sqrt_ps : GCCBuiltin<"__builtin_ia32_sqrtps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty],
                        [InstrNoMem]>;
  def int_x86_sse_rcp_ss : GCCBuiltin<"__builtin_ia32_rcpss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty],
                        [InstrNoMem]>;
  def int_x86_sse_rcp_ps : GCCBuiltin<"__builtin_ia32_rcpps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty],
                        [InstrNoMem]>;
  def int_x86_sse_rsqrt_ss : GCCBuiltin<"__builtin_ia32_rsqrtss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty],
                        [InstrNoMem]>;
  def int_x86_sse_rsqrt_ps : GCCBuiltin<"__builtin_ia32_rsqrtps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty],
                        [InstrNoMem]>;
  def int_x86_sse_min_ss : GCCBuiltin<"__builtin_ia32_minss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_min_ps : GCCBuiltin<"__builtin_ia32_minps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_max_ss : GCCBuiltin<"__builtin_ia32_maxss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_max_ps : GCCBuiltin<"__builtin_ia32_maxps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

// Comparison ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmp_ss :
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_sbyte_ty], [InstrNoMem]>;
  def int_x86_sse_cmp_ps :
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_sbyte_ty], [InstrNoMem]>;
}


// Conversion ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cvtss2si : GCCBuiltin<"__builtin_ia32_cvtss2si">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cvtps2pi : GCCBuiltin<"__builtin_ia32_cvtps2pi">,
              Intrinsic<[llvm_v2i32_ty, llvm_v4i32_ty], [InstrNoMem]>;
  def int_x86_sse_cvttss2si : GCCBuiltin<"__builtin_ia32_cvttss2si">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cvttps2pi : GCCBuiltin<"__builtin_ia32_cvttps2pi">,
              Intrinsic<[llvm_v2i32_ty, llvm_v4i32_ty], [InstrNoMem]>;
  def int_x86_sse_cvtsi2ss : GCCBuiltin<"__builtin_ia32_cvtsi2ss">,
              Intrinsic<[llvm_v4f32_ty, llvm_int_ty], [InstrNoMem]>;
  def int_x86_sse_cvtpi2ps : GCCBuiltin<"__builtin_ia32_cvtpi2ps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v2i32_ty], [InstrNoMem]>;
}

// SIMD load ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_loadh_ps : GCCBuiltin<"__builtin_ia32_loadhps">,
              Intrinsic<[llvm_v4f32_ty, llvm_ptr_ty], [IntrReadMem]>;
  def int_x86_sse_loadl_ps : GCCBuiltin<"__builtin_ia32_loadlps">,
              Intrinsic<[llvm_v4f32_ty, llvm_ptr_ty], [IntrReadMem]>;
  def int_x86_sse_loadu_ps : GCCBuiltin<"__builtin_ia32_loadups">,
              Intrinsic<[llvm_v4f32_ty, llvm_ptr_ty], [IntrReadMem]>;
}

// SIMD store ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_storeh_ps : GCCBuiltin<"__builtin_ia32_storehps">,
              Intrinsic<[llvm_ptr_ty, llvm_v4f32_ty], [IntrWriteMem]>;
  def int_x86_sse_storel_ps : GCCBuiltin<"__builtin_ia32_storelps">,
              Intrinsic<[llvm_ptr_ty, llvm_v4f32_ty], [IntrWriteMem]>;
  def int_x86_sse_storeu_ps : GCCBuiltin<"__builtin_ia32_storeups">,
              Intrinsic<[llvm_ptr_ty, llvm_v4f32_ty], [IntrWriteMem]>;
}

// Cacheability support ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_prefetch : GCCBuiltin<"__builtin_ia32_prefetch">,
              Intrinsic<[llvm_ptr_ty, llvm_int_ty], [IntrWriteMem]>;
  def int_x86_sse_movntq : GCCBuiltin<"__builtin_ia32_movntq">,
              Intrinsic<[llvm_ptr_ty, llvm_v2i32_ty], [IntrWriteMem]>;
  def int_x86_sse_movntps : GCCBuiltin<"__builtin_ia32_movntps">,
              Intrinsic<[llvm_ptr_ty, llvm_v4f32_ty], [IntrWriteMem]>;
  def int_x86_sse_sfence : GCCBuiltin<"__builtin_ia32_sfence">,
              Intrinsic<[llvm_void_ty], [IntrWriteMem]>;
}

// Control register.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_stmxcsr : GCCBuiltin<"__builtin_ia32_stmxcsr">,
              Intrinsic<[llvm_void_ty, llvm_ptr_ty], [IntrWriteMem]>;
  def int_x86_sse_ldmxcsr : GCCBuiltin<"__builtin_ia32_ldmxcsr">,
              Intrinsic<[llvm_void_ty, llvm_ptr_ty], [IntrWriteMem]>;
}

// Misc.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_movmsk_ps : GCCBuiltin<"__builtin_ia32_movmskps">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty], [InstrNoMem]>;
}

//===----------------------------------------------------------------------===//
// SSE2

// Arithmetic ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_add_sd : GCCBuiltin<"__builtin_ia32_addsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [InstrNoMem]>;
  def int_x86_sse2_sub_sd : GCCBuiltin<"__builtin_ia32_subsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [InstrNoMem]>;
  def int_x86_sse2_mul_sd : GCCBuiltin<"__builtin_ia32_mulsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [InstrNoMem]>;
  def int_x86_sse2_div_sd : GCCBuiltin<"__builtin_ia32_divsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [InstrNoMem]>;
  def int_x86_sse2_sqrt_sd : GCCBuiltin<"__builtin_ia32_sqrtsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty],
                        [InstrNoMem]>;
  def int_x86_sse2_sqrt_pd : GCCBuiltin<"__builtin_ia32_sqrtpd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty],
                        [InstrNoMem]>;
  def int_x86_sse2_rcp_sd : GCCBuiltin<"__builtin_ia32_rcpsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty],
                        [InstrNoMem]>;
  def int_x86_sse2_rcp_pd : GCCBuiltin<"__builtin_ia32_rcppd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty],
                        [InstrNoMem]>;
  def int_x86_sse2_rsqrt_sd : GCCBuiltin<"__builtin_ia32_rsqrtsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty],
                        [InstrNoMem]>;
  def int_x86_sse2_rsqrt_pd : GCCBuiltin<"__builtin_ia32_rsqrtpd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty],
                        [InstrNoMem]>;
  def int_x86_sse2_min_sd : GCCBuiltin<"__builtin_ia32_minsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [InstrNoMem]>;
  def int_x86_sse2_min_pd : GCCBuiltin<"__builtin_ia32_minpd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [InstrNoMem]>;
  def int_x86_sse2_max_sd : GCCBuiltin<"__builtin_ia32_maxsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [InstrNoMem]>;
  def int_x86_sse2_max_pd : GCCBuiltin<"__builtin_ia32_maxpd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [InstrNoMem]>;
}

// Integer shift ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_psll_dq : GCCBuiltin<"__builtin_ia32_pslldqi128">,
              Intrinsic<[llvm_v2i64_ty, llvm_v2i64_ty,
                         llvm_int_ty], [InstrNoMem]>;
  def int_x86_sse2_psrl_dq : GCCBuiltin<"__builtin_ia32_psrldqi128">,
              Intrinsic<[llvm_v2i64_ty, llvm_v2i64_ty,
                         llvm_int_ty], [InstrNoMem]>;
}

// Misc.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_packsswb_128 : GCCBuiltin<"__builtin_ia32_packsswb128">,
              Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty,
                         llvm_v8i16_ty], [InstrNoMem]>;
  def int_x86_sse2_packssdw_128 : GCCBuiltin<"__builtin_ia32_packssdw128">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty,
                         llvm_v4i32_ty], [InstrNoMem]>;
  def int_x86_sse2_packuswb_128 : GCCBuiltin<"__builtin_ia32_packuswb128">,
              Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty,
                         llvm_v8i16_ty], [InstrNoMem]>;
  def int_x86_sse2_movmskpd : GCCBuiltin<"__builtin_ia32_movmskpd">,
              Intrinsic<[llvm_int_ty, llvm_v2f64_ty], [InstrNoMem]>;
  def int_x86_sse2_pmovmskb_128 : GCCBuiltin<"__builtin_ia32_pmovmskb128">,
              Intrinsic<[llvm_int_ty, llvm_v16i8_ty], [InstrNoMem]>;
}

//===----------------------------------------------------------------------===//
// SSE3

// Horizontal ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse3_hadd_ps : GCCBuiltin<"__builtin_ia32_haddps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse3_hadd_pd : GCCBuiltin<"__builtin_ia32_haddpd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [InstrNoMem]>;
  def int_x86_sse3_hsub_ps : GCCBuiltin<"__builtin_ia32_hsubps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse3_hsub_pd : GCCBuiltin<"__builtin_ia32_hsubpd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [InstrNoMem]>;
}
