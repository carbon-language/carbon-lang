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
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_sub_ss : GCCBuiltin<"__builtin_ia32_subss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_mul_ss : GCCBuiltin<"__builtin_ia32_mulss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_div_ss : GCCBuiltin<"__builtin_ia32_divss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_sqrt_ss : GCCBuiltin<"__builtin_ia32_sqrtss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_sqrt_ps : GCCBuiltin<"__builtin_ia32_sqrtps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_rcp_ss : GCCBuiltin<"__builtin_ia32_rcpss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_rcp_ps : GCCBuiltin<"__builtin_ia32_rcpps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_rsqrt_ss : GCCBuiltin<"__builtin_ia32_rsqrtss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_rsqrt_ps : GCCBuiltin<"__builtin_ia32_rsqrtps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_min_ss : GCCBuiltin<"__builtin_ia32_minss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_min_ps : GCCBuiltin<"__builtin_ia32_minps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_max_ss : GCCBuiltin<"__builtin_ia32_maxss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_max_ps : GCCBuiltin<"__builtin_ia32_maxps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
}

// Comparison ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmp_ss :
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_sbyte_ty], [IntrNoMem]>;
  def int_x86_sse_cmp_ps :
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_sbyte_ty], [IntrNoMem]>;
  def int_x86_sse_comieq_ss : GCCBuiltin<"__builtin_ia32_comieq">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_comilt_ss : GCCBuiltin<"__builtin_ia32_comilt">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_comile_ss : GCCBuiltin<"__builtin_ia32_comile">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_comigt_ss : GCCBuiltin<"__builtin_ia32_comigt">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_comige_ss : GCCBuiltin<"__builtin_ia32_comige">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_comineq_ss : GCCBuiltin<"__builtin_ia32_comineq">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomieq_ss : GCCBuiltin<"__builtin_ia32_ucomieq">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomilt_ss : GCCBuiltin<"__builtin_ia32_ucomilt">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomile_ss : GCCBuiltin<"__builtin_ia32_ucomile">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomigt_ss : GCCBuiltin<"__builtin_ia32_ucomigt">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomige_ss : GCCBuiltin<"__builtin_ia32_ucomige">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomineq_ss : GCCBuiltin<"__builtin_ia32_ucomineq">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
}


// Conversion ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cvtss2si : GCCBuiltin<"__builtin_ia32_cvtss2si">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_cvtps2pi : GCCBuiltin<"__builtin_ia32_cvtps2pi">,
              Intrinsic<[llvm_v2i32_ty, llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_sse_cvttss2si : GCCBuiltin<"__builtin_ia32_cvttss2si">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_cvttps2pi : GCCBuiltin<"__builtin_ia32_cvttps2pi">,
              Intrinsic<[llvm_v2i32_ty, llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_sse_cvtsi2ss : GCCBuiltin<"__builtin_ia32_cvtsi2ss">,
              Intrinsic<[llvm_v4f32_ty, llvm_int_ty], [IntrNoMem]>;
  def int_x86_sse_cvtpi2ps : GCCBuiltin<"__builtin_ia32_cvtpi2ps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v2i32_ty], [IntrNoMem]>;
}

// SIMD load ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_loadu_ps : GCCBuiltin<"__builtin_ia32_loadups">,
              Intrinsic<[llvm_v4f32_ty, llvm_ptr_ty], [IntrReadMem]>;
}

// SIMD store ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_storeu_ps : GCCBuiltin<"__builtin_ia32_storeups">,
              Intrinsic<[llvm_void_ty, llvm_ptr_ty,
                         llvm_v4f32_ty], [IntrWriteMem]>;
}

// Cacheability support ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_movnt_ps : GCCBuiltin<"__builtin_ia32_movntps">,
              Intrinsic<[llvm_void_ty, llvm_ptr_ty,
                         llvm_v4f32_ty], [IntrWriteMem]>;
  def int_x86_sse_sfence : GCCBuiltin<"__builtin_ia32_sfence">,
              Intrinsic<[llvm_void_ty], [IntrWriteMem]>;
}

// Control register.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_stmxcsr :
              Intrinsic<[llvm_void_ty, llvm_ptr_ty], [IntrWriteMem]>;
  def int_x86_sse_ldmxcsr :
              Intrinsic<[llvm_void_ty, llvm_ptr_ty], [IntrWriteMem]>;
}

// Misc.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_movmsk_ps : GCCBuiltin<"__builtin_ia32_movmskps">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// SSE2

// FP arithmetic ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_add_sd : GCCBuiltin<"__builtin_ia32_addsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_sub_sd : GCCBuiltin<"__builtin_ia32_subsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_mul_sd : GCCBuiltin<"__builtin_ia32_mulsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_div_sd : GCCBuiltin<"__builtin_ia32_divsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_sqrt_sd : GCCBuiltin<"__builtin_ia32_sqrtsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_sse2_sqrt_pd : GCCBuiltin<"__builtin_ia32_sqrtpd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_sse2_rcp_sd : GCCBuiltin<"__builtin_ia32_rcpsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_sse2_rcp_pd : GCCBuiltin<"__builtin_ia32_rcppd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_sse2_rsqrt_sd : GCCBuiltin<"__builtin_ia32_rsqrtsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_sse2_rsqrt_pd : GCCBuiltin<"__builtin_ia32_rsqrtpd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_sse2_min_sd : GCCBuiltin<"__builtin_ia32_minsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_min_pd : GCCBuiltin<"__builtin_ia32_minpd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_max_sd : GCCBuiltin<"__builtin_ia32_maxsd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_max_pd : GCCBuiltin<"__builtin_ia32_maxpd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
}

// FP comparison ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_cmp_sd :
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty, llvm_sbyte_ty], [IntrNoMem]>;
  def int_x86_sse2_cmp_pd :
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty, llvm_sbyte_ty], [IntrNoMem]>;
  def int_x86_sse2_comieq_sd : GCCBuiltin<"__builtin_ia32_comisdeq">,
              Intrinsic<[llvm_int_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_comilt_sd : GCCBuiltin<"__builtin_ia32_comisdlt">,
              Intrinsic<[llvm_int_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_comile_sd : GCCBuiltin<"__builtin_ia32_comisdle">,
              Intrinsic<[llvm_int_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_comigt_sd : GCCBuiltin<"__builtin_ia32_comisdgt">,
              Intrinsic<[llvm_int_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_comige_sd : GCCBuiltin<"__builtin_ia32_comisdge">,
              Intrinsic<[llvm_int_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_comineq_sd : GCCBuiltin<"__builtin_ia32_comisdneq">,
              Intrinsic<[llvm_int_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomieq_sd : GCCBuiltin<"__builtin_ia32_ucomisdeq">,
              Intrinsic<[llvm_int_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomilt_sd : GCCBuiltin<"__builtin_ia32_ucomisdlt">,
              Intrinsic<[llvm_int_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomile_sd : GCCBuiltin<"__builtin_ia32_ucomisdle">,
              Intrinsic<[llvm_int_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomigt_sd : GCCBuiltin<"__builtin_ia32_ucomisdgt">,
              Intrinsic<[llvm_int_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomige_sd : GCCBuiltin<"__builtin_ia32_ucomisdge">,
              Intrinsic<[llvm_int_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomineq_sd : GCCBuiltin<"__builtin_ia32_ucomisdneq">,
              Intrinsic<[llvm_int_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
}

// Integer shift ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_psll_dq : GCCBuiltin<"__builtin_ia32_pslldqi128">,
              Intrinsic<[llvm_v2i64_ty, llvm_v2i64_ty,
                         llvm_int_ty], [IntrNoMem]>;
  def int_x86_sse2_psrl_dq : GCCBuiltin<"__builtin_ia32_psrldqi128">,
              Intrinsic<[llvm_v2i64_ty, llvm_v2i64_ty,
                         llvm_int_ty], [IntrNoMem]>;
}

// SIMD load ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_loadu_pd : GCCBuiltin<"__builtin_ia32_loadupd">,
              Intrinsic<[llvm_v2f64_ty, llvm_ptr_ty], [IntrReadMem]>;
  def int_x86_sse2_loadu_dq : GCCBuiltin<"__builtin_ia32_loaddqu">,
              Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty], [IntrReadMem]>;
}

// SIMD store ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_storeu_pd : GCCBuiltin<"__builtin_ia32_storeupd">,
              Intrinsic<[llvm_void_ty, llvm_ptr_ty,
                         llvm_v2f64_ty], [IntrWriteMem]>;
  def int_x86_sse2_storeu_dq : GCCBuiltin<"__builtin_ia32_storedqu">,
              Intrinsic<[llvm_void_ty, llvm_ptr_ty,
                         llvm_v16i8_ty], [IntrWriteMem]>;
  def int_x86_sse2_storel_dq : GCCBuiltin<"__builtin_ia32_storelv4si">,
              Intrinsic<[llvm_void_ty, llvm_ptr_ty,
                         llvm_v4i32_ty], [IntrWriteMem]>;
}

// Cacheability support ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_movnt_dq : GCCBuiltin<"__builtin_ia32_movntdq">,
              Intrinsic<[llvm_void_ty, llvm_ptr_ty,
                         llvm_v2i64_ty], [IntrWriteMem]>;
  def int_x86_sse2_movnt_pd : GCCBuiltin<"__builtin_ia32_movntpd">,
              Intrinsic<[llvm_void_ty, llvm_ptr_ty,
                         llvm_v2f64_ty], [IntrWriteMem]>;
  def int_x86_sse2_movnt_i : GCCBuiltin<"__builtin_ia32_movnti">,
              Intrinsic<[llvm_void_ty, llvm_ptr_ty,
                         llvm_int_ty], [IntrWriteMem]>;
}

// Misc.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_packsswb_128 : GCCBuiltin<"__builtin_ia32_packsswb128">,
              Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_sse2_packssdw_128 : GCCBuiltin<"__builtin_ia32_packssdw128">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_sse2_packuswb_128 : GCCBuiltin<"__builtin_ia32_packuswb128">,
              Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  // FIXME: Temporary workaround since 2-wide shuffle is broken.
  def int_x86_sse2_movl_dq : GCCBuiltin<"__builtin_ia32_movqv4si">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_sse2_movmskpd : GCCBuiltin<"__builtin_ia32_movmskpd">,
              Intrinsic<[llvm_int_ty, llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_pmovmskb_128 : GCCBuiltin<"__builtin_ia32_pmovmskb128">,
              Intrinsic<[llvm_int_ty, llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_sse2_maskmov_dqu : GCCBuiltin<"__builtin_ia32_maskmovdqu">,
              Intrinsic<[llvm_void_ty, llvm_v16i8_ty,
                         llvm_v16i8_ty, llvm_ptr_ty], [IntrWriteMem]>;
}

//===----------------------------------------------------------------------===//
// SSE3

// Horizontal ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse3_hadd_ps : GCCBuiltin<"__builtin_ia32_haddps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse3_hadd_pd : GCCBuiltin<"__builtin_ia32_haddpd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse3_hsub_ps : GCCBuiltin<"__builtin_ia32_hsubps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse3_hsub_pd : GCCBuiltin<"__builtin_ia32_hsubpd">,
              Intrinsic<[llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
}
