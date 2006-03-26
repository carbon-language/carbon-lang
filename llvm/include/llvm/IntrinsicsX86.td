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
  def int_x86_sse_addss : GCCBuiltin<"__builtin_ia32_addss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_addps : GCCBuiltin<"__builtin_ia32_addps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_subss : GCCBuiltin<"__builtin_ia32_subss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_subps : GCCBuiltin<"__builtin_ia32_subps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_mulss : GCCBuiltin<"__builtin_ia32_mulss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_mulps : GCCBuiltin<"__builtin_ia32_mulps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_divss : GCCBuiltin<"__builtin_ia32_divss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_divps : GCCBuiltin<"__builtin_ia32_divps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_sqrtss : GCCBuiltin<"__builtin_ia32_sqrtss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_sqrtps : GCCBuiltin<"__builtin_ia32_sqrtps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_rcpss : GCCBuiltin<"__builtin_ia32_rcpss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_rcpps : GCCBuiltin<"__builtin_ia32_rcpps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_rsqrtss : GCCBuiltin<"__builtin_ia32_rsqrtss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_rsqrtps : GCCBuiltin<"__builtin_ia32_rsqrtps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_minss : GCCBuiltin<"__builtin_ia32_minss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_minps : GCCBuiltin<"__builtin_ia32_minps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_maxss : GCCBuiltin<"__builtin_ia32_maxss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_maxps : GCCBuiltin<"__builtin_ia32_maxps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

// Logical ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_andps : GCCBuiltin<"__builtin_ia32_andps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_andnotps : GCCBuiltin<"__builtin_ia32_andnotps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_orps : GCCBuiltin<"__builtin_ia32_orps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_xorps : GCCBuiltin<"__builtin_ia32_xorps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

// Comparison ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpeqss : GCCBuiltin<"__builtin_ia32_cmpeqss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpeqps : GCCBuiltin<"__builtin_ia32_cmpeqps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpltss : GCCBuiltin<"__builtin_ia32_cmpltss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_cmpltps : GCCBuiltin<"__builtin_ia32_cmpltps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpless : GCCBuiltin<"__builtin_ia32_cmpless">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_cmpleps : GCCBuiltin<"__builtin_ia32_cmpleps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpgtss : GCCBuiltin<"__builtin_ia32_cmpgtss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_cmpgtps : GCCBuiltin<"__builtin_ia32_cmpgtps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpgess : GCCBuiltin<"__builtin_ia32_cmpgess">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_cmpgeps : GCCBuiltin<"__builtin_ia32_cmpgeps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpneqss : GCCBuiltin<"__builtin_ia32_cmpneqss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_cmpneqps : GCCBuiltin<"__builtin_ia32_cmpneqps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpnltss : GCCBuiltin<"__builtin_ia32_cmpnltss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_cmpnltps : GCCBuiltin<"__builtin_ia32_cmpnltps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpnless : GCCBuiltin<"__builtin_ia32_cmpnless">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_cmpnleps : GCCBuiltin<"__builtin_ia32_cmpnleps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpngtss : GCCBuiltin<"__builtin_ia32_cmpngtss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_cmpngtps : GCCBuiltin<"__builtin_ia32_cmpngtps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpngess : GCCBuiltin<"__builtin_ia32_cmpngess">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_cmpngeps : GCCBuiltin<"__builtin_ia32_cmpngeps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpordss : GCCBuiltin<"__builtin_ia32_cmpordss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_cmpordps : GCCBuiltin<"__builtin_ia32_cmpordps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_cmpunordss : GCCBuiltin<"__builtin_ia32_cmpunordss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_cmpunordps : GCCBuiltin<"__builtin_ia32_cmpunordps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_comieqss : GCCBuiltin<"__builtin_ia32_comieq">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_comiltss : GCCBuiltin<"__builtin_ia32_comilt">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_comiless : GCCBuiltin<"__Builtin_ia32_comile">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_comigtss : GCCBuiltin<"__builtin_ia32_comigt">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_comigess : GCCBuiltin<"__builtin_ia32_comige">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_comineqss : GCCBuiltin<"__builtin_ia32_comineq">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_ucomieqss : GCCBuiltin<"__builtin_ia32_ucomieq">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_ucomiltss : GCCBuiltin<"__builtin_ia32_ucomilt">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_ucomiless : GCCBuiltin<"__Builtin_ia32_ucomile">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_ucomigtss : GCCBuiltin<"__builtin_ia32_ucomigt">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_ucomigess : GCCBuiltin<"__builtin_ia32_ucomige">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_ucomineqss : GCCBuiltin<"__builtin_ia32_ucomineq">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}


// Conversion ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cvtss2si : GCCBuiltin<"__builtin_ia32_cvtss2si">,
              Intrinsic<[llvm_int_ty, llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_cvtps2pi : GCCBuiltin<"__builtin_ia32_cvtps2pi">,
              Intrinsic<[llvm_v2i32_ty, llvm_v4i32_ty], [InstrNoMem]>;
  def int_x86_sse_cvttss2si : GCCBuiltin<"__builtin_ia32_cvttss2si">,
              Intrinsic<[llvm_int_ty, llvm_float_ty], [InstrNoMem]>;
  def int_x86_sse_cvttps2pi : GCCBuiltin<"__builtin_ia32_cvttps2pi">,
              Intrinsic<[llvm_v2i32_ty, llvm_v4i32_ty], [InstrNoMem]>;
  def int_x86_sse_cvtsi2ss : GCCBuiltin<"__builtin_ia32_cvtsi2ss">,
              Intrinsic<[llvm_float_ty, llvm_int_ty], [InstrNoMem]>;
  def int_x86_sse_cvtpi2ps : GCCBuiltin<"__builtin_ia32_cvtpi2ps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v2i32_ty], [InstrNoMem]>;
}

// SIMD load ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_loadhps : GCCBuiltin<"__builtin_ia32_loadhps">,
              Intrinsic<[llvm_v4f32_ty, llvm_ptr_ty], [IntrReadMem]>;
  def int_x86_sse_loadlps : GCCBuiltin<"__builtin_ia32_loadlps">,
              Intrinsic<[llvm_v4f32_ty, llvm_ptr_ty], [IntrReadMem]>;
  def int_x86_sse_loadups : GCCBuiltin<"__builtin_ia32_loadups">,
              Intrinsic<[llvm_v4f32_ty, llvm_ptr_ty], [IntrReadMem]>;
}

// SIMD store ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_storehps : GCCBuiltin<"__builtin_ia32_storehps">,
              Intrinsic<[llvm_ptr_ty, llvm_v4f32_ty], [IntrWriteMem]>;
  def int_x86_sse_storelps : GCCBuiltin<"__builtin_ia32_storelps">,
              Intrinsic<[llvm_ptr_ty, llvm_v4f32_ty], [IntrWriteMem]>;
  def int_x86_sse_storeups : GCCBuiltin<"__builtin_ia32_storeups">,
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
  def int_x86_sse_movmskps : GCCBuiltin<"__builtin_ia32_movmskps">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty], [InstrNoMem]>;
  def int_x86_sse_ldmxcsr : GCCBuiltin<"__builtin_ia32_ldmxcsr">,
              Intrinsic<[llvm_void_ty, llvm_ptr_ty], [IntrWriteMem]>;
}

//===----------------------------------------------------------------------===//
// SSE2

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_movmskpd : GCCBuiltin<"__builtin_ia32_movmskpd">,
              Intrinsic<[llvm_int_ty, llvm_v2f64_ty], [InstrNoMem]>;
}
