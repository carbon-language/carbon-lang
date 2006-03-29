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
  def int_x86_sse_cmpeq_ss : GCCBuiltin<"__builtin_ia32_cmpeqss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpeq_ps : GCCBuiltin<"__builtin_ia32_cmpeqps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmplt_ss : GCCBuiltin<"__builtin_ia32_cmpltss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmplt_ps : GCCBuiltin<"__builtin_ia32_cmpltps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmple_ss : GCCBuiltin<"__builtin_ia32_cmpless">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmple_ps : GCCBuiltin<"__builtin_ia32_cmpleps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpgt_ss : GCCBuiltin<"__builtin_ia32_cmpgtss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpgt_ps : GCCBuiltin<"__builtin_ia32_cmpgtps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpge_ss : GCCBuiltin<"__builtin_ia32_cmpgess">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpge_ps : GCCBuiltin<"__builtin_ia32_cmpgeps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpneq_ss : GCCBuiltin<"__builtin_ia32_cmpneqss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpneq_ps : GCCBuiltin<"__builtin_ia32_cmpneqps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpnlt_ss : GCCBuiltin<"__builtin_ia32_cmpnltss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpnlt_ps : GCCBuiltin<"__builtin_ia32_cmpnltps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpnle_ss : GCCBuiltin<"__builtin_ia32_cmpnless">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpnle_ps : GCCBuiltin<"__builtin_ia32_cmpnleps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpngt_ss : GCCBuiltin<"__builtin_ia32_cmpngtss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpngt_ps : GCCBuiltin<"__builtin_ia32_cmpngtps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpnge_ss : GCCBuiltin<"__builtin_ia32_cmpngess">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpnge_ps : GCCBuiltin<"__builtin_ia32_cmpngeps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpord_ss : GCCBuiltin<"__builtin_ia32_cmpordss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpord_ps : GCCBuiltin<"__builtin_ia32_cmpordps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpunord_ss : GCCBuiltin<"__builtin_ia32_cmpunordss">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpunord_ps : GCCBuiltin<"__builtin_ia32_cmpunordps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_comieq_ss : GCCBuiltin<"__builtin_ia32_comieq">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_comilt_ss : GCCBuiltin<"__builtin_ia32_comilt">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_comile_ss : GCCBuiltin<"__Builtin_ia32_comile">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_comigt_ss : GCCBuiltin<"__builtin_ia32_comigt">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_comige_ss : GCCBuiltin<"__builtin_ia32_comige">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_comineq_ss : GCCBuiltin<"__builtin_ia32_comineq">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_ucomieq_ss : GCCBuiltin<"__builtin_ia32_ucomieq">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_ucomilt_ss : GCCBuiltin<"__builtin_ia32_ucomilt">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_ucomile_ss : GCCBuiltin<"__Builtin_ia32_ucomile">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_ucomigt_ss : GCCBuiltin<"__builtin_ia32_ucomigt">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_ucomige_ss : GCCBuiltin<"__builtin_ia32_ucomige">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_ucomineq_ss : GCCBuiltin<"__builtin_ia32_ucomineq">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
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
}
