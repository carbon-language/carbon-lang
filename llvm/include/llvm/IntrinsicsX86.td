//===- IntrinsicsX86.td - Defines X86 intrinsics -----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the X86-specific intrinsics.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Interrupt traps
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_int : Intrinsic<[], [llvm_i8_ty]>;
}

//===----------------------------------------------------------------------===//
// 3DNow!

let TargetPrefix = "x86" in {
  def int_x86_3dnow_pavgusb : GCCBuiltin<"__builtin_ia32_pavgusb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pf2id : GCCBuiltin<"__builtin_ia32_pf2id">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_3dnow_pfacc : GCCBuiltin<"__builtin_ia32_pfacc">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfadd : GCCBuiltin<"__builtin_ia32_pfadd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfcmpeq : GCCBuiltin<"__builtin_ia32_pfcmpeq">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfcmpge : GCCBuiltin<"__builtin_ia32_pfcmpge">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfcmpgt : GCCBuiltin<"__builtin_ia32_pfcmpgt">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfmax : GCCBuiltin<"__builtin_ia32_pfmax">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfmin : GCCBuiltin<"__builtin_ia32_pfmin">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfmul : GCCBuiltin<"__builtin_ia32_pfmul">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfrcp : GCCBuiltin<"__builtin_ia32_pfrcp">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_3dnow_pfrcpit1 : GCCBuiltin<"__builtin_ia32_pfrcpit1">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfrcpit2 : GCCBuiltin<"__builtin_ia32_pfrcpit2">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfrsqrt : GCCBuiltin<"__builtin_ia32_pfrsqrt">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_3dnow_pfrsqit1 : GCCBuiltin<"__builtin_ia32_pfrsqit1">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfsub : GCCBuiltin<"__builtin_ia32_pfsub">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfsubr : GCCBuiltin<"__builtin_ia32_pfsubr">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pi2fd : GCCBuiltin<"__builtin_ia32_pi2fd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_3dnow_pmulhrw : GCCBuiltin<"__builtin_ia32_pmulhrw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// 3DNow! extensions

let TargetPrefix = "x86" in {
  def int_x86_3dnowa_pf2iw : GCCBuiltin<"__builtin_ia32_pf2iw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_3dnowa_pfnacc : GCCBuiltin<"__builtin_ia32_pfnacc">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnowa_pfpnacc : GCCBuiltin<"__builtin_ia32_pfpnacc">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnowa_pi2fw : GCCBuiltin<"__builtin_ia32_pi2fw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_3dnowa_pswapd :
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// SSE1

// Arithmetic ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_add_ss : GCCBuiltin<"__builtin_ia32_addss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_sub_ss : GCCBuiltin<"__builtin_ia32_subss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_mul_ss : GCCBuiltin<"__builtin_ia32_mulss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_div_ss : GCCBuiltin<"__builtin_ia32_divss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_sqrt_ss : GCCBuiltin<"__builtin_ia32_sqrtss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_sqrt_ps : GCCBuiltin<"__builtin_ia32_sqrtps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_rcp_ss : GCCBuiltin<"__builtin_ia32_rcpss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_rcp_ps : GCCBuiltin<"__builtin_ia32_rcpps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_rsqrt_ss : GCCBuiltin<"__builtin_ia32_rsqrtss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_rsqrt_ps : GCCBuiltin<"__builtin_ia32_rsqrtps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_min_ss : GCCBuiltin<"__builtin_ia32_minss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_min_ps : GCCBuiltin<"__builtin_ia32_minps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_max_ss : GCCBuiltin<"__builtin_ia32_maxss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_max_ps : GCCBuiltin<"__builtin_ia32_maxps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
}

// Comparison ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmp_ss :
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_sse_cmp_ps :
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_sse_comieq_ss : GCCBuiltin<"__builtin_ia32_comieq">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_comilt_ss : GCCBuiltin<"__builtin_ia32_comilt">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_comile_ss : GCCBuiltin<"__builtin_ia32_comile">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_comigt_ss : GCCBuiltin<"__builtin_ia32_comigt">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_comige_ss : GCCBuiltin<"__builtin_ia32_comige">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_comineq_ss : GCCBuiltin<"__builtin_ia32_comineq">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomieq_ss : GCCBuiltin<"__builtin_ia32_ucomieq">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomilt_ss : GCCBuiltin<"__builtin_ia32_ucomilt">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomile_ss : GCCBuiltin<"__builtin_ia32_ucomile">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomigt_ss : GCCBuiltin<"__builtin_ia32_ucomigt">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomige_ss : GCCBuiltin<"__builtin_ia32_ucomige">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomineq_ss : GCCBuiltin<"__builtin_ia32_ucomineq">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
}


// Conversion ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cvtss2si : GCCBuiltin<"__builtin_ia32_cvtss2si">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_cvtss2si64 : GCCBuiltin<"__builtin_ia32_cvtss2si64">,
              Intrinsic<[llvm_i64_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_cvttss2si : GCCBuiltin<"__builtin_ia32_cvttss2si">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_cvttss2si64 : GCCBuiltin<"__builtin_ia32_cvttss2si64">,
              Intrinsic<[llvm_i64_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_cvtsi2ss : GCCBuiltin<"__builtin_ia32_cvtsi2ss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse_cvtsi642ss : GCCBuiltin<"__builtin_ia32_cvtsi642ss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_i64_ty], [IntrNoMem]>;
  def int_x86_sse_cvtps2pi : GCCBuiltin<"__builtin_ia32_cvtps2pi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_cvttps2pi: GCCBuiltin<"__builtin_ia32_cvttps2pi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_cvtpi2ps : GCCBuiltin<"__builtin_ia32_cvtpi2ps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
}

// SIMD store ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_storeu_ps : GCCBuiltin<"__builtin_ia32_storeups">,
              Intrinsic<[], [llvm_ptr_ty,
                         llvm_v4f32_ty], []>;
}

// Cacheability support ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_sfence : GCCBuiltin<"__builtin_ia32_sfence">,
              Intrinsic<[], [], []>;
}

// Control register.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_stmxcsr :
              Intrinsic<[], [llvm_ptr_ty], []>;
  def int_x86_sse_ldmxcsr :
              Intrinsic<[], [llvm_ptr_ty], []>;
}

// Misc.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_movmsk_ps : GCCBuiltin<"__builtin_ia32_movmskps">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// SSE2

// FP arithmetic ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_add_sd : GCCBuiltin<"__builtin_ia32_addsd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_sub_sd : GCCBuiltin<"__builtin_ia32_subsd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_mul_sd : GCCBuiltin<"__builtin_ia32_mulsd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_div_sd : GCCBuiltin<"__builtin_ia32_divsd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_sqrt_sd : GCCBuiltin<"__builtin_ia32_sqrtsd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_sse2_sqrt_pd : GCCBuiltin<"__builtin_ia32_sqrtpd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_sse2_min_sd : GCCBuiltin<"__builtin_ia32_minsd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_min_pd : GCCBuiltin<"__builtin_ia32_minpd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_max_sd : GCCBuiltin<"__builtin_ia32_maxsd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_max_pd : GCCBuiltin<"__builtin_ia32_maxpd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
}

// FP comparison ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_cmp_sd :
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_sse2_cmp_pd :
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_sse2_comieq_sd : GCCBuiltin<"__builtin_ia32_comisdeq">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_comilt_sd : GCCBuiltin<"__builtin_ia32_comisdlt">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_comile_sd : GCCBuiltin<"__builtin_ia32_comisdle">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_comigt_sd : GCCBuiltin<"__builtin_ia32_comisdgt">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_comige_sd : GCCBuiltin<"__builtin_ia32_comisdge">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_comineq_sd : GCCBuiltin<"__builtin_ia32_comisdneq">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomieq_sd : GCCBuiltin<"__builtin_ia32_ucomisdeq">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomilt_sd : GCCBuiltin<"__builtin_ia32_ucomisdlt">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomile_sd : GCCBuiltin<"__builtin_ia32_ucomisdle">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomigt_sd : GCCBuiltin<"__builtin_ia32_ucomisdgt">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomige_sd : GCCBuiltin<"__builtin_ia32_ucomisdge">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomineq_sd : GCCBuiltin<"__builtin_ia32_ucomisdneq">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
}

// Integer arithmetic ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_padds_b : GCCBuiltin<"__builtin_ia32_paddsb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_padds_w : GCCBuiltin<"__builtin_ia32_paddsw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_paddus_b : GCCBuiltin<"__builtin_ia32_paddusb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_paddus_w : GCCBuiltin<"__builtin_ia32_paddusw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_psubs_b : GCCBuiltin<"__builtin_ia32_psubsb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_sse2_psubs_w : GCCBuiltin<"__builtin_ia32_psubsw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_sse2_psubus_b : GCCBuiltin<"__builtin_ia32_psubusb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_sse2_psubus_w : GCCBuiltin<"__builtin_ia32_psubusw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_sse2_pmulhu_w : GCCBuiltin<"__builtin_ia32_pmulhuw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_pmulh_w : GCCBuiltin<"__builtin_ia32_pmulhw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_pmulu_dq : GCCBuiltin<"__builtin_ia32_pmuludq128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_pmadd_wd : GCCBuiltin<"__builtin_ia32_pmaddwd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_pavg_b : GCCBuiltin<"__builtin_ia32_pavgb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_pavg_w : GCCBuiltin<"__builtin_ia32_pavgw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_pmaxu_b : GCCBuiltin<"__builtin_ia32_pmaxub128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_pmaxs_w : GCCBuiltin<"__builtin_ia32_pmaxsw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_pminu_b : GCCBuiltin<"__builtin_ia32_pminub128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_pmins_w : GCCBuiltin<"__builtin_ia32_pminsw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_psad_bw : GCCBuiltin<"__builtin_ia32_psadbw128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem, Commutative]>;
}

// Integer shift ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_psll_w : GCCBuiltin<"__builtin_ia32_psllw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_sse2_psll_d : GCCBuiltin<"__builtin_ia32_pslld128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_sse2_psll_q : GCCBuiltin<"__builtin_ia32_psllq128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                         llvm_v2i64_ty], [IntrNoMem]>;
  def int_x86_sse2_psrl_w : GCCBuiltin<"__builtin_ia32_psrlw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_sse2_psrl_d : GCCBuiltin<"__builtin_ia32_psrld128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_sse2_psrl_q : GCCBuiltin<"__builtin_ia32_psrlq128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                         llvm_v2i64_ty], [IntrNoMem]>;
  def int_x86_sse2_psra_w : GCCBuiltin<"__builtin_ia32_psraw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_sse2_psra_d : GCCBuiltin<"__builtin_ia32_psrad128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;

  def int_x86_sse2_pslli_w : GCCBuiltin<"__builtin_ia32_psllwi128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_pslli_d : GCCBuiltin<"__builtin_ia32_pslldi128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_pslli_q : GCCBuiltin<"__builtin_ia32_psllqi128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_psrli_w : GCCBuiltin<"__builtin_ia32_psrlwi128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_psrli_d : GCCBuiltin<"__builtin_ia32_psrldi128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_psrli_q : GCCBuiltin<"__builtin_ia32_psrlqi128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_psrai_w : GCCBuiltin<"__builtin_ia32_psrawi128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_psrai_d : GCCBuiltin<"__builtin_ia32_psradi128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_i32_ty], [IntrNoMem]>;

  def int_x86_sse2_psll_dq : GCCBuiltin<"__builtin_ia32_pslldqi128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_psrl_dq : GCCBuiltin<"__builtin_ia32_psrldqi128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_psll_dq_bs : GCCBuiltin<"__builtin_ia32_pslldqi128_byteshift">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_psrl_dq_bs : GCCBuiltin<"__builtin_ia32_psrldqi128_byteshift">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
}

// Integer comparison ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_pcmpeq_b : GCCBuiltin<"__builtin_ia32_pcmpeqb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_pcmpeq_w : GCCBuiltin<"__builtin_ia32_pcmpeqw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_pcmpeq_d : GCCBuiltin<"__builtin_ia32_pcmpeqd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_pcmpgt_b : GCCBuiltin<"__builtin_ia32_pcmpgtb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_sse2_pcmpgt_w : GCCBuiltin<"__builtin_ia32_pcmpgtw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_sse2_pcmpgt_d : GCCBuiltin<"__builtin_ia32_pcmpgtd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;
}

// Conversion ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_cvtdq2pd : GCCBuiltin<"__builtin_ia32_cvtdq2pd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtdq2ps : GCCBuiltin<"__builtin_ia32_cvtdq2ps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtpd2dq : GCCBuiltin<"__builtin_ia32_cvtpd2dq">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvttpd2dq : GCCBuiltin<"__builtin_ia32_cvttpd2dq">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtpd2ps : GCCBuiltin<"__builtin_ia32_cvtpd2ps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtps2dq : GCCBuiltin<"__builtin_ia32_cvtps2dq">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse2_cvttps2dq : GCCBuiltin<"__builtin_ia32_cvttps2dq">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtps2pd : GCCBuiltin<"__builtin_ia32_cvtps2pd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtsd2si : GCCBuiltin<"__builtin_ia32_cvtsd2si">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtsd2si64 : GCCBuiltin<"__builtin_ia32_cvtsd2si64">,
              Intrinsic<[llvm_i64_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvttsd2si : GCCBuiltin<"__builtin_ia32_cvttsd2si">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvttsd2si64 : GCCBuiltin<"__builtin_ia32_cvttsd2si64">,
              Intrinsic<[llvm_i64_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtsi2sd : GCCBuiltin<"__builtin_ia32_cvtsi2sd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtsi642sd : GCCBuiltin<"__builtin_ia32_cvtsi642sd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_i64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtsd2ss : GCCBuiltin<"__builtin_ia32_cvtsd2ss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtss2sd : GCCBuiltin<"__builtin_ia32_cvtss2sd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_cvtpd2pi : GCCBuiltin<"__builtin_ia32_cvtpd2pi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse_cvttpd2pi: GCCBuiltin<"__builtin_ia32_cvttpd2pi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse_cvtpi2pd : GCCBuiltin<"__builtin_ia32_cvtpi2pd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
}

// SIMD store ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_storeu_pd : GCCBuiltin<"__builtin_ia32_storeupd">,
              Intrinsic<[], [llvm_ptr_ty,
                         llvm_v2f64_ty], []>;
  def int_x86_sse2_storeu_dq : GCCBuiltin<"__builtin_ia32_storedqu">,
              Intrinsic<[], [llvm_ptr_ty,
                         llvm_v16i8_ty], []>;
  def int_x86_sse2_storel_dq : GCCBuiltin<"__builtin_ia32_storelv4si">,
              Intrinsic<[], [llvm_ptr_ty,
                         llvm_v4i32_ty], []>;
}

// Misc.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_packsswb_128 : GCCBuiltin<"__builtin_ia32_packsswb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_sse2_packssdw_128 : GCCBuiltin<"__builtin_ia32_packssdw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_sse2_packuswb_128 : GCCBuiltin<"__builtin_ia32_packuswb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_sse2_movmsk_pd : GCCBuiltin<"__builtin_ia32_movmskpd">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_pmovmskb_128 : GCCBuiltin<"__builtin_ia32_pmovmskb128">,
              Intrinsic<[llvm_i32_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_sse2_maskmov_dqu : GCCBuiltin<"__builtin_ia32_maskmovdqu">,
              Intrinsic<[], [llvm_v16i8_ty,
                         llvm_v16i8_ty, llvm_ptr_ty], []>;
  def int_x86_sse2_clflush : GCCBuiltin<"__builtin_ia32_clflush">,
              Intrinsic<[], [llvm_ptr_ty], []>;
  def int_x86_sse2_lfence : GCCBuiltin<"__builtin_ia32_lfence">,
              Intrinsic<[], [], []>;
  def int_x86_sse2_mfence : GCCBuiltin<"__builtin_ia32_mfence">,
              Intrinsic<[], [], []>;
}

//===----------------------------------------------------------------------===//
// SSE3

// Addition / subtraction ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse3_addsub_ps : GCCBuiltin<"__builtin_ia32_addsubps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse3_addsub_pd : GCCBuiltin<"__builtin_ia32_addsubpd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
}

// Horizontal ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse3_hadd_ps : GCCBuiltin<"__builtin_ia32_haddps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse3_hadd_pd : GCCBuiltin<"__builtin_ia32_haddpd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse3_hsub_ps : GCCBuiltin<"__builtin_ia32_hsubps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse3_hsub_pd : GCCBuiltin<"__builtin_ia32_hsubpd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
}

// Specialized unaligned load.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse3_ldu_dq : GCCBuiltin<"__builtin_ia32_lddqu">,
              Intrinsic<[llvm_v16i8_ty], [llvm_ptr_ty], [IntrReadMem]>;
}

// Thread synchronization ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse3_monitor : GCCBuiltin<"__builtin_ia32_monitor">,
              Intrinsic<[], [llvm_ptr_ty,
                         llvm_i32_ty, llvm_i32_ty], []>;
  def int_x86_sse3_mwait : GCCBuiltin<"__builtin_ia32_mwait">,
              Intrinsic<[], [llvm_i32_ty,
                         llvm_i32_ty], []>;
}

//===----------------------------------------------------------------------===//
// SSSE3

// Horizontal arithmetic ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_ssse3_phadd_w         : GCCBuiltin<"__builtin_ia32_phaddw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_phadd_w_128     : GCCBuiltin<"__builtin_ia32_phaddw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;

  def int_x86_ssse3_phadd_d         : GCCBuiltin<"__builtin_ia32_phaddd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_phadd_d_128     : GCCBuiltin<"__builtin_ia32_phaddd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;

  def int_x86_ssse3_phadd_sw        : GCCBuiltin<"__builtin_ia32_phaddsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_phadd_sw_128    : GCCBuiltin<"__builtin_ia32_phaddsw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;

  def int_x86_ssse3_phsub_w         : GCCBuiltin<"__builtin_ia32_phsubw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_phsub_w_128     : GCCBuiltin<"__builtin_ia32_phsubw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;

  def int_x86_ssse3_phsub_d         : GCCBuiltin<"__builtin_ia32_phsubd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_phsub_d_128     : GCCBuiltin<"__builtin_ia32_phsubd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;

  def int_x86_ssse3_phsub_sw        : GCCBuiltin<"__builtin_ia32_phsubsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_phsub_sw_128    : GCCBuiltin<"__builtin_ia32_phsubsw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;

  def int_x86_ssse3_pmadd_ub_sw     : GCCBuiltin<"__builtin_ia32_pmaddubsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_pmadd_ub_sw_128 : GCCBuiltin<"__builtin_ia32_pmaddubsw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
}

// Packed multiply high with round and scale
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_ssse3_pmul_hr_sw      : GCCBuiltin<"__builtin_ia32_pmulhrsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_ssse3_pmul_hr_sw_128  : GCCBuiltin<"__builtin_ia32_pmulhrsw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem, Commutative]>;
}

// Shuffle ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_ssse3_pshuf_b         : GCCBuiltin<"__builtin_ia32_pshufb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_pshuf_b_128     : GCCBuiltin<"__builtin_ia32_pshufb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_sse_pshuf_w           : GCCBuiltin<"__builtin_ia32_pshufw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_i8_ty],
                         [IntrNoMem]>;
}

// Sign ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_ssse3_psign_b         : GCCBuiltin<"__builtin_ia32_psignb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_psign_b_128     : GCCBuiltin<"__builtin_ia32_psignb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem]>;

  def int_x86_ssse3_psign_w         : GCCBuiltin<"__builtin_ia32_psignw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_psign_w_128     : GCCBuiltin<"__builtin_ia32_psignw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;

  def int_x86_ssse3_psign_d         : GCCBuiltin<"__builtin_ia32_psignd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_psign_d_128     : GCCBuiltin<"__builtin_ia32_psignd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;
}

// Absolute value ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_ssse3_pabs_b     : GCCBuiltin<"__builtin_ia32_pabsb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_pabs_b_128 : GCCBuiltin<"__builtin_ia32_pabsb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty], [IntrNoMem]>;

  def int_x86_ssse3_pabs_w     : GCCBuiltin<"__builtin_ia32_pabsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_pabs_w_128 : GCCBuiltin<"__builtin_ia32_pabsw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty], [IntrNoMem]>;

  def int_x86_ssse3_pabs_d     : GCCBuiltin<"__builtin_ia32_pabsd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_pabs_d_128 : GCCBuiltin<"__builtin_ia32_pabsd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// SSE4.1

// FP rounding ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_round_ss        : GCCBuiltin<"__builtin_ia32_roundss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse41_round_ps        : GCCBuiltin<"__builtin_ia32_roundps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse41_round_sd        : GCCBuiltin<"__builtin_ia32_roundsd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse41_round_pd        : GCCBuiltin<"__builtin_ia32_roundpd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
}

// Vector sign and zero extend
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_pmovsxbd        : GCCBuiltin<"__builtin_ia32_pmovsxbd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_x86_sse41_pmovsxbq        : GCCBuiltin<"__builtin_ia32_pmovsxbq128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_x86_sse41_pmovsxbw        : GCCBuiltin<"__builtin_ia32_pmovsxbw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_x86_sse41_pmovsxdq        : GCCBuiltin<"__builtin_ia32_pmovsxdq128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_x86_sse41_pmovsxwd        : GCCBuiltin<"__builtin_ia32_pmovsxwd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_x86_sse41_pmovsxwq        : GCCBuiltin<"__builtin_ia32_pmovsxwq128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_x86_sse41_pmovzxbd        : GCCBuiltin<"__builtin_ia32_pmovzxbd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_x86_sse41_pmovzxbq        : GCCBuiltin<"__builtin_ia32_pmovzxbq128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_x86_sse41_pmovzxbw        : GCCBuiltin<"__builtin_ia32_pmovzxbw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_x86_sse41_pmovzxdq        : GCCBuiltin<"__builtin_ia32_pmovzxdq128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_x86_sse41_pmovzxwd        : GCCBuiltin<"__builtin_ia32_pmovzxwd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_x86_sse41_pmovzxwq        : GCCBuiltin<"__builtin_ia32_pmovzxwq128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v8i16_ty],
                        [IntrNoMem]>;
}

// Vector min element
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_phminposuw     : GCCBuiltin<"__builtin_ia32_phminposuw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty],
                        [IntrNoMem]>;
}

// Vector compare, min, max
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_pcmpeqq         : GCCBuiltin<"__builtin_ia32_pcmpeqq">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem, Commutative]>;
  def int_x86_sse42_pcmpgtq         : GCCBuiltin<"__builtin_ia32_pcmpgtq">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_sse41_pmaxsb          : GCCBuiltin<"__builtin_ia32_pmaxsb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                        [IntrNoMem, Commutative]>;
  def int_x86_sse41_pmaxsd          : GCCBuiltin<"__builtin_ia32_pmaxsd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem, Commutative]>;
  def int_x86_sse41_pmaxud          : GCCBuiltin<"__builtin_ia32_pmaxud128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem, Commutative]>;
  def int_x86_sse41_pmaxuw          : GCCBuiltin<"__builtin_ia32_pmaxuw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                        [IntrNoMem, Commutative]>;
  def int_x86_sse41_pminsb          : GCCBuiltin<"__builtin_ia32_pminsb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                        [IntrNoMem, Commutative]>;
  def int_x86_sse41_pminsd          : GCCBuiltin<"__builtin_ia32_pminsd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem, Commutative]>;
  def int_x86_sse41_pminud          : GCCBuiltin<"__builtin_ia32_pminud128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem, Commutative]>;
  def int_x86_sse41_pminuw          : GCCBuiltin<"__builtin_ia32_pminuw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                        [IntrNoMem, Commutative]>;
}

// Advanced Encryption Standard (AES) Instructions
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_aesni_aesimc          : GCCBuiltin<"__builtin_ia32_aesimc128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_aesni_aesenc          : GCCBuiltin<"__builtin_ia32_aesenc128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_aesni_aesenclast : GCCBuiltin<"__builtin_ia32_aesenclast128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_aesni_aesdec          : GCCBuiltin<"__builtin_ia32_aesdec128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_aesni_aesdeclast : GCCBuiltin<"__builtin_ia32_aesdeclast128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_aesni_aeskeygenassist :
              GCCBuiltin<"__builtin_ia32_aeskeygenassist128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_i8_ty],
                        [IntrNoMem]>;
}

// Vector pack
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_packusdw        : GCCBuiltin<"__builtin_ia32_packusdw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
}

// Vector multiply
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_pmuldq          : GCCBuiltin<"__builtin_ia32_pmuldq128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem, Commutative]>;
}

// Vector extract
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_pextrb         :
              Intrinsic<[llvm_i32_ty], [llvm_v16i8_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_x86_sse41_pextrd         :
              Intrinsic<[llvm_i32_ty], [llvm_v4i32_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_x86_sse41_pextrq         :
              Intrinsic<[llvm_i64_ty], [llvm_v2i64_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_x86_sse41_extractps      : GCCBuiltin<"__builtin_ia32_extractps128">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty, llvm_i32_ty],
                        [IntrNoMem]>;
}

// Vector insert
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_insertps       : GCCBuiltin<"__builtin_ia32_insertps128">,
          Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,llvm_i32_ty],
                    [IntrNoMem]>;
}

// Vector blend
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_pblendvb         : GCCBuiltin<"__builtin_ia32_pblendvb128">,
        Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty,llvm_v16i8_ty],
                  [IntrNoMem]>;
  def int_x86_sse41_pblendw          : GCCBuiltin<"__builtin_ia32_pblendw128">,
        Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty],
                  [IntrNoMem]>;
  def int_x86_sse41_blendpd          : GCCBuiltin<"__builtin_ia32_blendpd">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty, llvm_i32_ty],
                  [IntrNoMem]>;
  def int_x86_sse41_blendps          : GCCBuiltin<"__builtin_ia32_blendps">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty, llvm_i32_ty],
                  [IntrNoMem]>;
  def int_x86_sse41_blendvpd         : GCCBuiltin<"__builtin_ia32_blendvpd">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,llvm_v2f64_ty],
                  [IntrNoMem]>;
  def int_x86_sse41_blendvps         : GCCBuiltin<"__builtin_ia32_blendvps">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,llvm_v4f32_ty],
                  [IntrNoMem]>;
}

// Vector dot product
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_dppd            : GCCBuiltin<"__builtin_ia32_dppd">,
          Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,llvm_i32_ty],
                    [IntrNoMem, Commutative]>;
  def int_x86_sse41_dpps            : GCCBuiltin<"__builtin_ia32_dpps">,
          Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,llvm_i32_ty],
                    [IntrNoMem, Commutative]>;
}

// Vector sum of absolute differences
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_mpsadbw         : GCCBuiltin<"__builtin_ia32_mpsadbw128">,
          Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty, llvm_v16i8_ty,llvm_i32_ty],
                    [IntrNoMem, Commutative]>;
}

// Cacheability support ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_movntdqa        : GCCBuiltin<"__builtin_ia32_movntdqa">,
          Intrinsic<[llvm_v2i64_ty], [llvm_ptr_ty], [IntrReadMem]>;
}

// Test instruction with bitwise comparison.
let TargetPrefix = "x86" in { // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_ptestz          : GCCBuiltin<"__builtin_ia32_ptestz128">,
          Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty, llvm_v4f32_ty],
                    [IntrNoMem]>;
  def int_x86_sse41_ptestc          : GCCBuiltin<"__builtin_ia32_ptestc128">,
          Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty, llvm_v4f32_ty],
                    [IntrNoMem]>;
  def int_x86_sse41_ptestnzc        : GCCBuiltin<"__builtin_ia32_ptestnzc128">,
          Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty, llvm_v4f32_ty],
                    [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// SSE4.2

// Miscellaneous
// CRC Instruction
let TargetPrefix = "x86" in { // All intrinsics start with "llvm.x86.".
  def int_x86_sse42_crc32_32_8       : GCCBuiltin<"__builtin_ia32_crc32qi">,
          Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_sse42_crc32_32_16      : GCCBuiltin<"__builtin_ia32_crc32hi">,
          Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i16_ty],
                    [IntrNoMem]>;
  def int_x86_sse42_crc32_32_32      : GCCBuiltin<"__builtin_ia32_crc32si">,
          Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                    [IntrNoMem]>;
  def int_x86_sse42_crc32_64_8       :
          Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_sse42_crc32_64_64      : GCCBuiltin<"__builtin_ia32_crc32di">,
          Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
                    [IntrNoMem]>;
}

// String/text processing ops.
let TargetPrefix = "x86" in { // All intrinsics start with "llvm.x86.".
  def int_x86_sse42_pcmpistrm128  : GCCBuiltin<"__builtin_ia32_pcmpistrm128">,
    Intrinsic<[llvm_v16i8_ty],
        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpistri128  : GCCBuiltin<"__builtin_ia32_pcmpistri128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpistria128 : GCCBuiltin<"__builtin_ia32_pcmpistria128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpistric128 : GCCBuiltin<"__builtin_ia32_pcmpistric128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpistrio128 : GCCBuiltin<"__builtin_ia32_pcmpistrio128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpistris128 : GCCBuiltin<"__builtin_ia32_pcmpistris128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpistriz128 : GCCBuiltin<"__builtin_ia32_pcmpistriz128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpestrm128  : GCCBuiltin<"__builtin_ia32_pcmpestrm128">,
    Intrinsic<[llvm_v16i8_ty],
        [llvm_v16i8_ty, llvm_i32_ty, llvm_v16i8_ty, llvm_i32_ty,
         llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpestri128  : GCCBuiltin<"__builtin_ia32_pcmpestri128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_i32_ty, llvm_v16i8_ty, llvm_i32_ty,
         llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpestria128 : GCCBuiltin<"__builtin_ia32_pcmpestria128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_i32_ty, llvm_v16i8_ty, llvm_i32_ty,
         llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpestric128 : GCCBuiltin<"__builtin_ia32_pcmpestric128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_i32_ty, llvm_v16i8_ty, llvm_i32_ty,
         llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpestrio128 : GCCBuiltin<"__builtin_ia32_pcmpestrio128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_i32_ty, llvm_v16i8_ty, llvm_i32_ty,
         llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpestris128 : GCCBuiltin<"__builtin_ia32_pcmpestris128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_i32_ty, llvm_v16i8_ty, llvm_i32_ty,
         llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpestriz128 : GCCBuiltin<"__builtin_ia32_pcmpestriz128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_i32_ty, llvm_v16i8_ty, llvm_i32_ty,
         llvm_i8_ty],
        [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// AVX

// Arithmetic ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_addsub_pd_256 : GCCBuiltin<"__builtin_ia32_addsubpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_addsub_ps_256 : GCCBuiltin<"__builtin_ia32_addsubps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty], [IntrNoMem]>;
  def int_x86_avx_max_pd_256 : GCCBuiltin<"__builtin_ia32_maxpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_max_ps_256 : GCCBuiltin<"__builtin_ia32_maxps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty], [IntrNoMem]>;
  def int_x86_avx_min_pd_256 : GCCBuiltin<"__builtin_ia32_minpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_min_ps_256 : GCCBuiltin<"__builtin_ia32_minps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty], [IntrNoMem]>;

  def int_x86_avx_sqrt_pd_256 : GCCBuiltin<"__builtin_ia32_sqrtpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_sqrt_ps_256 : GCCBuiltin<"__builtin_ia32_sqrtps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty], [IntrNoMem]>;

  def int_x86_avx_rsqrt_ps_256 : GCCBuiltin<"__builtin_ia32_rsqrtps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty], [IntrNoMem]>;

  def int_x86_avx_rcp_ps_256 : GCCBuiltin<"__builtin_ia32_rcpps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty], [IntrNoMem]>;

  def int_x86_avx_round_pd_256 : GCCBuiltin<"__builtin_ia32_roundpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx_round_ps_256 : GCCBuiltin<"__builtin_ia32_roundps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_i32_ty], [IntrNoMem]>;
}

// Horizontal ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_hadd_pd_256 : GCCBuiltin<"__builtin_ia32_haddpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_hsub_ps_256 : GCCBuiltin<"__builtin_ia32_hsubps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty], [IntrNoMem]>;
  def int_x86_avx_hsub_pd_256 : GCCBuiltin<"__builtin_ia32_hsubpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_hadd_ps_256 : GCCBuiltin<"__builtin_ia32_haddps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty], [IntrNoMem]>;
}

// Vector permutation
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_vpermilvar_pd : GCCBuiltin<"__builtin_ia32_vpermilvarpd">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                  llvm_v2i64_ty], [IntrNoMem]>;
  def int_x86_avx_vpermilvar_ps : GCCBuiltin<"__builtin_ia32_vpermilvarps">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                  llvm_v4i32_ty], [IntrNoMem]>;

  def int_x86_avx_vpermilvar_pd_256 :
        GCCBuiltin<"__builtin_ia32_vpermilvarpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty, llvm_v4i64_ty], [IntrNoMem]>;
  def int_x86_avx_vpermilvar_ps_256 :
        GCCBuiltin<"__builtin_ia32_vpermilvarps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty, llvm_v8i32_ty], [IntrNoMem]>;

  def int_x86_avx_vperm2f128_pd_256 :
        GCCBuiltin<"__builtin_ia32_vperm2f128_pd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx_vperm2f128_ps_256 :
        GCCBuiltin<"__builtin_ia32_vperm2f128_ps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx_vperm2f128_si_256 :
        GCCBuiltin<"__builtin_ia32_vperm2f128_si256">,
        Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                  llvm_v8i32_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx_vpermil_pd : GCCBuiltin<"__builtin_ia32_vpermilpd">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                  llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx_vpermil_ps : GCCBuiltin<"__builtin_ia32_vpermilps">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                  llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx_vpermil_pd_256 : GCCBuiltin<"__builtin_ia32_vpermilpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx_vpermil_ps_256 : GCCBuiltin<"__builtin_ia32_vpermilps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_i8_ty], [IntrNoMem]>;
}

// Vector blend
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_blend_pd_256 : GCCBuiltin<"__builtin_ia32_blendpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx_blend_ps_256 : GCCBuiltin<"__builtin_ia32_blendps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx_blendv_pd_256 : GCCBuiltin<"__builtin_ia32_blendvpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty, llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_blendv_ps_256 : GCCBuiltin<"__builtin_ia32_blendvps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty, llvm_v8f32_ty], [IntrNoMem]>;
}

// Vector dot product
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_dp_ps_256 : GCCBuiltin<"__builtin_ia32_dpps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty, llvm_i32_ty], [IntrNoMem]>;
}

// Vector compare
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_cmp_pd_256 : GCCBuiltin<"__builtin_ia32_cmppd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx_cmp_ps_256 : GCCBuiltin<"__builtin_ia32_cmpps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty, llvm_i8_ty], [IntrNoMem]>;
}

// Vector extract and insert
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_vextractf128_pd_256 :
        GCCBuiltin<"__builtin_ia32_vextractf128_pd256">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v4f64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx_vextractf128_ps_256 :
        GCCBuiltin<"__builtin_ia32_vextractf128_ps256">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v8f32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx_vextractf128_si_256 :
        GCCBuiltin<"__builtin_ia32_vextractf128_si256">,
        Intrinsic<[llvm_v4i32_ty], [llvm_v8i32_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx_vinsertf128_pd_256 :
        GCCBuiltin<"__builtin_ia32_vinsertf128_pd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v2f64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx_vinsertf128_ps_256 :
        GCCBuiltin<"__builtin_ia32_vinsertf128_ps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v4f32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx_vinsertf128_si_256 :
        GCCBuiltin<"__builtin_ia32_vinsertf128_si256">,
        Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                  llvm_v4i32_ty, llvm_i8_ty], [IntrNoMem]>;
}

// Vector convert
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_cvtdq2_pd_256 : GCCBuiltin<"__builtin_ia32_cvtdq2pd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_avx_cvtdq2_ps_256 : GCCBuiltin<"__builtin_ia32_cvtdq2ps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8i32_ty], [IntrNoMem]>;
  def int_x86_avx_cvt_pd2_ps_256 : GCCBuiltin<"__builtin_ia32_cvtpd2ps256">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_cvt_ps2dq_256 : GCCBuiltin<"__builtin_ia32_cvtps2dq256">,
        Intrinsic<[llvm_v8i32_ty], [llvm_v8f32_ty], [IntrNoMem]>;
  def int_x86_avx_cvt_ps2_pd_256 : GCCBuiltin<"__builtin_ia32_cvtps2pd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_avx_cvtt_pd2dq_256 : GCCBuiltin<"__builtin_ia32_cvttpd2dq256">,
        Intrinsic<[llvm_v4i32_ty], [llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_cvt_pd2dq_256 : GCCBuiltin<"__builtin_ia32_cvtpd2dq256">,
        Intrinsic<[llvm_v4i32_ty], [llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_cvtt_ps2dq_256 : GCCBuiltin<"__builtin_ia32_cvttps2dq256">,
        Intrinsic<[llvm_v8i32_ty], [llvm_v8f32_ty], [IntrNoMem]>;
}

// Vector bit test
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_vtestz_pd : GCCBuiltin<"__builtin_ia32_vtestzpd">,
        Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                  llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_avx_vtestc_pd : GCCBuiltin<"__builtin_ia32_vtestcpd">,
        Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                  llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_avx_vtestnzc_pd : GCCBuiltin<"__builtin_ia32_vtestnzcpd">,
        Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                  llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_avx_vtestz_ps : GCCBuiltin<"__builtin_ia32_vtestzps">,
        Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                  llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_avx_vtestc_ps : GCCBuiltin<"__builtin_ia32_vtestcps">,
        Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                  llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_avx_vtestnzc_ps : GCCBuiltin<"__builtin_ia32_vtestnzcps">,
        Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                  llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_avx_vtestz_pd_256 : GCCBuiltin<"__builtin_ia32_vtestzpd256">,
        Intrinsic<[llvm_i32_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_vtestc_pd_256 : GCCBuiltin<"__builtin_ia32_vtestcpd256">,
        Intrinsic<[llvm_i32_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_vtestnzc_pd_256 : GCCBuiltin<"__builtin_ia32_vtestnzcpd256">,
        Intrinsic<[llvm_i32_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_vtestz_ps_256 : GCCBuiltin<"__builtin_ia32_vtestzps256">,
        Intrinsic<[llvm_i32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty], [IntrNoMem]>;
  def int_x86_avx_vtestc_ps_256 : GCCBuiltin<"__builtin_ia32_vtestcps256">,
        Intrinsic<[llvm_i32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty], [IntrNoMem]>;
  def int_x86_avx_vtestnzc_ps_256 : GCCBuiltin<"__builtin_ia32_vtestnzcps256">,
        Intrinsic<[llvm_i32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty], [IntrNoMem]>;
  def int_x86_avx_ptestz_256 : GCCBuiltin<"__builtin_ia32_ptestz256">,
        Intrinsic<[llvm_i32_ty], [llvm_v4i64_ty,
                  llvm_v4i64_ty], [IntrNoMem]>;
  def int_x86_avx_ptestc_256 : GCCBuiltin<"__builtin_ia32_ptestc256">,
        Intrinsic<[llvm_i32_ty], [llvm_v4i64_ty,
                  llvm_v4i64_ty], [IntrNoMem]>;
  def int_x86_avx_ptestnzc_256 : GCCBuiltin<"__builtin_ia32_ptestnzc256">,
        Intrinsic<[llvm_i32_ty], [llvm_v4i64_ty,
                  llvm_v4i64_ty], [IntrNoMem]>;
}

// Vector extract sign mask
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_movmsk_pd_256 : GCCBuiltin<"__builtin_ia32_movmskpd256">,
        Intrinsic<[llvm_i32_ty], [llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_movmsk_ps_256 : GCCBuiltin<"__builtin_ia32_movmskps256">,
        Intrinsic<[llvm_i32_ty], [llvm_v8f32_ty], [IntrNoMem]>;
}

// Vector zero
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_vzeroall : GCCBuiltin<"__builtin_ia32_vzeroall">,
        Intrinsic<[], [], []>;
  def int_x86_avx_vzeroupper : GCCBuiltin<"__builtin_ia32_vzeroupper">,
        Intrinsic<[], [], []>;
}

// Vector load with broadcast
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_vbroadcastss :
        GCCBuiltin<"__builtin_ia32_vbroadcastss">,
        Intrinsic<[llvm_v4f32_ty], [llvm_ptr_ty], [IntrReadMem]>;
  def int_x86_avx_vbroadcast_sd_256 :
        GCCBuiltin<"__builtin_ia32_vbroadcastsd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_ptr_ty], [IntrReadMem]>;
  def int_x86_avx_vbroadcastss_256 :
        GCCBuiltin<"__builtin_ia32_vbroadcastss256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_ptr_ty], [IntrReadMem]>;
  def int_x86_avx_vbroadcastf128_pd_256 :
        GCCBuiltin<"__builtin_ia32_vbroadcastf128_pd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_ptr_ty], [IntrReadMem]>;
  def int_x86_avx_vbroadcastf128_ps_256 :
        GCCBuiltin<"__builtin_ia32_vbroadcastf128_ps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_ptr_ty], [IntrReadMem]>;
}

// SIMD load ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_loadu_pd_256 : GCCBuiltin<"__builtin_ia32_loadupd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_ptr_ty], [IntrReadMem]>;
  def int_x86_avx_loadu_ps_256 : GCCBuiltin<"__builtin_ia32_loadups256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_ptr_ty], [IntrReadMem]>;
  def int_x86_avx_loadu_dq_256 : GCCBuiltin<"__builtin_ia32_loaddqu256">,
        Intrinsic<[llvm_v32i8_ty], [llvm_ptr_ty], [IntrReadMem]>;
  def int_x86_avx_ldu_dq_256 : GCCBuiltin<"__builtin_ia32_lddqu256">,
        Intrinsic<[llvm_v32i8_ty], [llvm_ptr_ty], [IntrReadMem]>;
}

// SIMD store ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_storeu_pd_256 : GCCBuiltin<"__builtin_ia32_storeupd256">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v4f64_ty], []>;
  def int_x86_avx_storeu_ps_256 : GCCBuiltin<"__builtin_ia32_storeups256">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v8f32_ty], []>;
  def int_x86_avx_storeu_dq_256 : GCCBuiltin<"__builtin_ia32_storedqu256">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v32i8_ty], []>;
}

// Cacheability support ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_movnt_dq_256 : GCCBuiltin<"__builtin_ia32_movntdq256">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v4i64_ty], []>;
  def int_x86_avx_movnt_pd_256 : GCCBuiltin<"__builtin_ia32_movntpd256">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v4f64_ty], []>;
  def int_x86_avx_movnt_ps_256 : GCCBuiltin<"__builtin_ia32_movntps256">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v8f32_ty], []>;
}

// Conditional load ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_maskload_pd : GCCBuiltin<"__builtin_ia32_maskloadpd">,
        Intrinsic<[llvm_v2f64_ty], [llvm_ptr_ty, llvm_v2f64_ty], [IntrReadMem]>;
  def int_x86_avx_maskload_ps : GCCBuiltin<"__builtin_ia32_maskloadps">,
        Intrinsic<[llvm_v4f32_ty], [llvm_ptr_ty, llvm_v4f32_ty], [IntrReadMem]>;
  def int_x86_avx_maskload_pd_256 : GCCBuiltin<"__builtin_ia32_maskloadpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_ptr_ty, llvm_v4f64_ty], [IntrReadMem]>;
  def int_x86_avx_maskload_ps_256 : GCCBuiltin<"__builtin_ia32_maskloadps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_ptr_ty, llvm_v8f32_ty], [IntrReadMem]>;
}

// Conditional store ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_maskstore_pd : GCCBuiltin<"__builtin_ia32_maskstorepd">,
        Intrinsic<[], [llvm_ptr_ty,
                  llvm_v2f64_ty, llvm_v2f64_ty], []>;
  def int_x86_avx_maskstore_ps : GCCBuiltin<"__builtin_ia32_maskstoreps">,
        Intrinsic<[], [llvm_ptr_ty,
                  llvm_v4f32_ty, llvm_v4f32_ty], []>;
  def int_x86_avx_maskstore_pd_256 :
        GCCBuiltin<"__builtin_ia32_maskstorepd256">,
        Intrinsic<[], [llvm_ptr_ty,
                  llvm_v4f64_ty, llvm_v4f64_ty], []>;
  def int_x86_avx_maskstore_ps_256 :
        GCCBuiltin<"__builtin_ia32_maskstoreps256">,
        Intrinsic<[], [llvm_ptr_ty,
                  llvm_v8f32_ty, llvm_v8f32_ty], []>;
}

//===----------------------------------------------------------------------===//
// AVX2

// Integer arithmetic ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_padds_b : GCCBuiltin<"__builtin_ia32_paddsb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_padds_w : GCCBuiltin<"__builtin_ia32_paddsw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_paddus_b : GCCBuiltin<"__builtin_ia32_paddusb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_paddus_w : GCCBuiltin<"__builtin_ia32_paddusw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_psubs_b : GCCBuiltin<"__builtin_ia32_psubsb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem]>;
  def int_x86_avx2_psubs_w : GCCBuiltin<"__builtin_ia32_psubsw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem]>;
  def int_x86_avx2_psubus_b : GCCBuiltin<"__builtin_ia32_psubusb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem]>;
  def int_x86_avx2_psubus_w : GCCBuiltin<"__builtin_ia32_psubusw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem]>;
  def int_x86_avx2_pmulhu_w : GCCBuiltin<"__builtin_ia32_pmulhuw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_pmulh_w : GCCBuiltin<"__builtin_ia32_pmulhw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_pmulu_dq : GCCBuiltin<"__builtin_ia32_pmuludq256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v8i32_ty,
                         llvm_v8i32_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_pmadd_wd : GCCBuiltin<"__builtin_ia32_pmaddwd256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_pavg_b : GCCBuiltin<"__builtin_ia32_pavgb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_pavg_w : GCCBuiltin<"__builtin_ia32_pavgw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_pmaxu_b : GCCBuiltin<"__builtin_ia32_pmaxub256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_pmaxs_w : GCCBuiltin<"__builtin_ia32_pmaxsw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_pminu_b : GCCBuiltin<"__builtin_ia32_pminub256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_pmins_w : GCCBuiltin<"__builtin_ia32_pminsw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_psad_bw : GCCBuiltin<"__builtin_ia32_psadbw256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem, Commutative]>;
}

// Integer shift ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_psll_w : GCCBuiltin<"__builtin_ia32_psllw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_avx2_psll_d : GCCBuiltin<"__builtin_ia32_pslld256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_avx2_psll_q : GCCBuiltin<"__builtin_ia32_psllq256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_v2i64_ty], [IntrNoMem]>;
  def int_x86_avx2_psrl_w : GCCBuiltin<"__builtin_ia32_psrlw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_avx2_psrl_d : GCCBuiltin<"__builtin_ia32_psrld256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_avx2_psrl_q : GCCBuiltin<"__builtin_ia32_psrlq256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_v2i64_ty], [IntrNoMem]>;
  def int_x86_avx2_psra_w : GCCBuiltin<"__builtin_ia32_psraw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_avx2_psra_d : GCCBuiltin<"__builtin_ia32_psrad256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;

  def int_x86_avx2_pslli_w : GCCBuiltin<"__builtin_ia32_psllwi256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx2_pslli_d : GCCBuiltin<"__builtin_ia32_pslldi256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx2_pslli_q : GCCBuiltin<"__builtin_ia32_psllqi256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx2_psrli_w : GCCBuiltin<"__builtin_ia32_psrlwi256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx2_psrli_d : GCCBuiltin<"__builtin_ia32_psrldi256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx2_psrli_q : GCCBuiltin<"__builtin_ia32_psrlqi256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx2_psrai_w : GCCBuiltin<"__builtin_ia32_psrawi256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx2_psrai_d : GCCBuiltin<"__builtin_ia32_psradi256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx2_psll_dq : GCCBuiltin<"__builtin_ia32_pslldqi256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx2_psrl_dq : GCCBuiltin<"__builtin_ia32_psrldqi256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx2_psll_dq_bs : GCCBuiltin<"__builtin_ia32_pslldqi256_byteshift">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx2_psrl_dq_bs : GCCBuiltin<"__builtin_ia32_psrldqi256_byteshift">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
}

// Integer comparison ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_pcmpeq_b : GCCBuiltin<"__builtin_ia32_pcmpeqb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty, llvm_v32i8_ty],
                        [IntrNoMem, Commutative]>;
  def int_x86_avx2_pcmpeq_w : GCCBuiltin<"__builtin_ia32_pcmpeqw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty, llvm_v16i16_ty],
                        [IntrNoMem, Commutative]>;
  def int_x86_avx2_pcmpeq_d : GCCBuiltin<"__builtin_ia32_pcmpeqd256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty, llvm_v8i32_ty],
                        [IntrNoMem, Commutative]>;
  def int_x86_avx2_pcmpgt_b : GCCBuiltin<"__builtin_ia32_pcmpgtb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem]>;
  def int_x86_avx2_pcmpgt_w : GCCBuiltin<"__builtin_ia32_pcmpgtw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem]>;
  def int_x86_avx2_pcmpgt_d : GCCBuiltin<"__builtin_ia32_pcmpgtd256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_v8i32_ty], [IntrNoMem]>;
}

// Pack ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_packsswb : GCCBuiltin<"__builtin_ia32_packsswb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem]>;
  def int_x86_avx2_packssdw : GCCBuiltin<"__builtin_ia32_packssdw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v8i32_ty,
                         llvm_v8i32_ty], [IntrNoMem]>;
  def int_x86_avx2_packuswb : GCCBuiltin<"__builtin_ia32_packuswb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// MMX

// Empty MMX state op.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_mmx_emms  : GCCBuiltin<"__builtin_ia32_emms">,
              Intrinsic<[], [], []>;
  def int_x86_mmx_femms : GCCBuiltin<"__builtin_ia32_femms">,
              Intrinsic<[], [], []>;
}

// Integer arithmetic ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  // Addition
  def int_x86_mmx_padd_b : GCCBuiltin<"__builtin_ia32_paddb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_padd_w : GCCBuiltin<"__builtin_ia32_paddw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_padd_d : GCCBuiltin<"__builtin_ia32_paddd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_padd_q : GCCBuiltin<"__builtin_ia32_paddq">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;

  def int_x86_mmx_padds_b : GCCBuiltin<"__builtin_ia32_paddsb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_padds_w : GCCBuiltin<"__builtin_ia32_paddsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;

  def int_x86_mmx_paddus_b : GCCBuiltin<"__builtin_ia32_paddusb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_paddus_w : GCCBuiltin<"__builtin_ia32_paddusw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;

  // Subtraction
  def int_x86_mmx_psub_b : GCCBuiltin<"__builtin_ia32_psubb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_psub_w : GCCBuiltin<"__builtin_ia32_psubw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_psub_d : GCCBuiltin<"__builtin_ia32_psubd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_psub_q : GCCBuiltin<"__builtin_ia32_psubq">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;

  def int_x86_mmx_psubs_b : GCCBuiltin<"__builtin_ia32_psubsb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_psubs_w : GCCBuiltin<"__builtin_ia32_psubsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;

  def int_x86_mmx_psubus_b : GCCBuiltin<"__builtin_ia32_psubusb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_psubus_w : GCCBuiltin<"__builtin_ia32_psubusw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;

  // Multiplication
  def int_x86_mmx_pmulh_w : GCCBuiltin<"__builtin_ia32_pmulhw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pmull_w : GCCBuiltin<"__builtin_ia32_pmullw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pmulhu_w : GCCBuiltin<"__builtin_ia32_pmulhuw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pmulu_dq : GCCBuiltin<"__builtin_ia32_pmuludq">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pmadd_wd : GCCBuiltin<"__builtin_ia32_pmaddwd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;

  // Bitwise operations
  def int_x86_mmx_pand : GCCBuiltin<"__builtin_ia32_pand">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_pandn : GCCBuiltin<"__builtin_ia32_pandn">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_por : GCCBuiltin<"__builtin_ia32_por">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_pxor : GCCBuiltin<"__builtin_ia32_pxor">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;

  // Averages
  def int_x86_mmx_pavg_b : GCCBuiltin<"__builtin_ia32_pavgb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pavg_w : GCCBuiltin<"__builtin_ia32_pavgw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;

  // Maximum
  def int_x86_mmx_pmaxu_b : GCCBuiltin<"__builtin_ia32_pmaxub">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pmaxs_w : GCCBuiltin<"__builtin_ia32_pmaxsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;

  // Minimum
  def int_x86_mmx_pminu_b : GCCBuiltin<"__builtin_ia32_pminub">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pmins_w : GCCBuiltin<"__builtin_ia32_pminsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;

  // Packed sum of absolute differences
  def int_x86_mmx_psad_bw : GCCBuiltin<"__builtin_ia32_psadbw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
}

// Integer shift ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  // Shift left logical
  def int_x86_mmx_psll_w : GCCBuiltin<"__builtin_ia32_psllw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_psll_d : GCCBuiltin<"__builtin_ia32_pslld">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_psll_q : GCCBuiltin<"__builtin_ia32_psllq">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;

  def int_x86_mmx_psrl_w : GCCBuiltin<"__builtin_ia32_psrlw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_psrl_d : GCCBuiltin<"__builtin_ia32_psrld">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_psrl_q : GCCBuiltin<"__builtin_ia32_psrlq">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;

  def int_x86_mmx_psra_w : GCCBuiltin<"__builtin_ia32_psraw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_psra_d : GCCBuiltin<"__builtin_ia32_psrad">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;

  def int_x86_mmx_pslli_w : GCCBuiltin<"__builtin_ia32_psllwi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_mmx_pslli_d : GCCBuiltin<"__builtin_ia32_pslldi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_mmx_pslli_q : GCCBuiltin<"__builtin_ia32_psllqi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_i32_ty], [IntrNoMem]>;

  def int_x86_mmx_psrli_w : GCCBuiltin<"__builtin_ia32_psrlwi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_mmx_psrli_d : GCCBuiltin<"__builtin_ia32_psrldi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_mmx_psrli_q : GCCBuiltin<"__builtin_ia32_psrlqi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_i32_ty], [IntrNoMem]>;

  def int_x86_mmx_psrai_w : GCCBuiltin<"__builtin_ia32_psrawi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_mmx_psrai_d : GCCBuiltin<"__builtin_ia32_psradi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_i32_ty], [IntrNoMem]>;
}

// Pack ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_mmx_packsswb : GCCBuiltin<"__builtin_ia32_packsswb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_packssdw : GCCBuiltin<"__builtin_ia32_packssdw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_packuswb : GCCBuiltin<"__builtin_ia32_packuswb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
}

// Unpacking ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_mmx_punpckhbw : GCCBuiltin<"__builtin_ia32_punpckhbw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_punpckhwd : GCCBuiltin<"__builtin_ia32_punpckhwd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_punpckhdq : GCCBuiltin<"__builtin_ia32_punpckhdq">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_punpcklbw : GCCBuiltin<"__builtin_ia32_punpcklbw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_punpcklwd : GCCBuiltin<"__builtin_ia32_punpcklwd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_punpckldq : GCCBuiltin<"__builtin_ia32_punpckldq">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
}

// Integer comparison ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_mmx_pcmpeq_b : GCCBuiltin<"__builtin_ia32_pcmpeqb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pcmpeq_w : GCCBuiltin<"__builtin_ia32_pcmpeqw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pcmpeq_d : GCCBuiltin<"__builtin_ia32_pcmpeqd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;

  def int_x86_mmx_pcmpgt_b : GCCBuiltin<"__builtin_ia32_pcmpgtb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_pcmpgt_w : GCCBuiltin<"__builtin_ia32_pcmpgtw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_pcmpgt_d : GCCBuiltin<"__builtin_ia32_pcmpgtd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
}

// Misc.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_mmx_maskmovq : GCCBuiltin<"__builtin_ia32_maskmovq">,
              Intrinsic<[], [llvm_x86mmx_ty, llvm_x86mmx_ty, llvm_ptr_ty], []>;

  def int_x86_mmx_pmovmskb : GCCBuiltin<"__builtin_ia32_pmovmskb">,
              Intrinsic<[llvm_i32_ty], [llvm_x86mmx_ty], [IntrNoMem]>;

  def int_x86_mmx_movnt_dq : GCCBuiltin<"__builtin_ia32_movntq">,
              Intrinsic<[], [llvm_ptrx86mmx_ty, llvm_x86mmx_ty], []>;

  def int_x86_mmx_palignr_b : GCCBuiltin<"__builtin_ia32_palignr">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                        llvm_x86mmx_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_mmx_pextr_w : GCCBuiltin<"__builtin_ia32_vec_ext_v4hi">,
              Intrinsic<[llvm_i32_ty], [llvm_x86mmx_ty, llvm_i32_ty],
                        [IntrNoMem]>;

  def int_x86_mmx_pinsr_w : GCCBuiltin<"__builtin_ia32_vec_set_v4hi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                        llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// BMI

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_bmi_bextr_32 : GCCBuiltin<"__builtin_ia32_bextr_u32">,
              Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_bmi_bextr_64 : GCCBuiltin<"__builtin_ia32_bextr_u64">,
              Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], [IntrNoMem]>;
  def int_x86_bmi_bzhi_32 : GCCBuiltin<"__builtin_ia32_bzhi_si">,
              Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_bmi_bzhi_64 : GCCBuiltin<"__builtin_ia32_bzhi_di">,
              Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], [IntrNoMem]>;
  def int_x86_bmi_pdep_32 : GCCBuiltin<"__builtin_ia32_pdep_si">,
              Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_bmi_pdep_64 : GCCBuiltin<"__builtin_ia32_pdep_di">,
              Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], [IntrNoMem]>;
  def int_x86_bmi_pext_32 : GCCBuiltin<"__builtin_ia32_pext_si">,
              Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_bmi_pext_64 : GCCBuiltin<"__builtin_ia32_pext_di">,
              Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// FS/GS Base

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_rdfsbase_32 : GCCBuiltin<"__builtin_ia32_rdfsbase32">,
              Intrinsic<[llvm_i32_ty], []>;
  def int_x86_rdgsbase_32 : GCCBuiltin<"__builtin_ia32_rdgsbase32">,
              Intrinsic<[llvm_i32_ty], []>;
  def int_x86_rdfsbase_64 : GCCBuiltin<"__builtin_ia32_rdfsbase64">,
              Intrinsic<[llvm_i64_ty], []>;
  def int_x86_rdgsbase_64 : GCCBuiltin<"__builtin_ia32_rdgsbase64">,
              Intrinsic<[llvm_i64_ty], []>;
  def int_x86_wrfsbase_32 : GCCBuiltin<"__builtin_ia32_wrfsbase32">,
              Intrinsic<[], [llvm_i32_ty]>;
  def int_x86_wrgsbase_32 : GCCBuiltin<"__builtin_ia32_wrgsbase32">,
              Intrinsic<[], [llvm_i32_ty]>;
  def int_x86_wrfsbase_64 : GCCBuiltin<"__builtin_ia32_wrfsbase64">,
              Intrinsic<[], [llvm_i64_ty]>;
  def int_x86_wrgsbase_64 : GCCBuiltin<"__builtin_ia32_wrgsbase64">,
              Intrinsic<[], [llvm_i64_ty]>;
}
