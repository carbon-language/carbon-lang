//===- IntrinsicsPowerPC.td - Defines PowerPC intrinsics ---*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines all of the PowerPC-specific intrinsics.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "ppc" in {  // All intrinsics start with "llvm.ppc.".
  def int_ppc_altivec_lvx : GCCBuiltin<"__builtin_altivec_lvx">,
              Intrinsic<[llvm_v4i32_ty, llvm_int_ty, llvm_ptr_ty],
                        [IntrReadMem]>;
  def int_ppc_altivec_stvx : GCCBuiltin<"__builtin_altivec_stvx">,
              Intrinsic<[llvm_void_ty, llvm_v4i32_ty, llvm_int_ty, llvm_ptr_ty],
                        [IntrWriteMem]>;
                            
  def int_ppc_altivec_vmaddfp : GCCBuiltin<"__builtin_altivec_vmaddfp">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_v4f32_ty], [InstrNoMem]>;
  def int_ppc_altivec_vnmsubfp : GCCBuiltin<"__builtin_altivec_vnmsubfp">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_v4f32_ty], [InstrNoMem]>;

  def int_ppc_altivec_vaddcuw : GCCBuiltin<"__builtin_altivec_vaddcuw">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vsubcuw : GCCBuiltin<"__builtin_altivec_vsubcuw">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vsel : GCCBuiltin<"__builtin_altivec_vsel_4si">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, 
                         llvm_v4i32_ty, llvm_v4i32_ty], [InstrNoMem]>;
  def int_ppc_altivec_vsldoi : GCCBuiltin<"__builtin_altivec_vsldoi_4si">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, 
                         llvm_v4i32_ty, llvm_int_ty], [InstrNoMem]>;

  // Comparisons setting a vector.
  def int_ppc_altivec_vcmpbfp : GCCBuiltin<"__builtin_altivec_vcmpbfp">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4f32_ty, llvm_v4f32_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpeqfp : GCCBuiltin<"__builtin_altivec_vcmpeqfp">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4f32_ty, llvm_v4f32_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpgefp : GCCBuiltin<"__builtin_altivec_vcmpgefp">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4f32_ty, llvm_v4f32_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpgtfp : GCCBuiltin<"__builtin_altivec_vcmpgtfp">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4f32_ty, llvm_v4f32_ty],
                        [InstrNoMem]>;
                        
  def int_ppc_altivec_vcmpequw : GCCBuiltin<"__builtin_altivec_vcmpequw">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpgtsw : GCCBuiltin<"__builtin_altivec_vcmpgtsw">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpgtuw : GCCBuiltin<"__builtin_altivec_vcmpgtuw">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;
                        
  def int_ppc_altivec_vcmpequh : GCCBuiltin<"__builtin_altivec_vcmpequh">,
              Intrinsic<[llvm_v4i32_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpgtsh : GCCBuiltin<"__builtin_altivec_vcmpgtsh">,
              Intrinsic<[llvm_v4i32_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpgtuh : GCCBuiltin<"__builtin_altivec_vcmpgtuh">,
              Intrinsic<[llvm_v4i32_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                        [InstrNoMem]>;

  def int_ppc_altivec_vcmpequb : GCCBuiltin<"__builtin_altivec_vcmpequb">,
              Intrinsic<[llvm_v4i32_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpgtsb : GCCBuiltin<"__builtin_altivec_vcmpgtsb">,
              Intrinsic<[llvm_v4i32_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpgtub : GCCBuiltin<"__builtin_altivec_vcmpgtub">,
              Intrinsic<[llvm_v4i32_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                        [InstrNoMem]>;

  // Saturating adds and subs.
  def int_ppc_altivec_vaddubs : GCCBuiltin<"__builtin_altivec_vaddubs">,
              Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vaddsbs : GCCBuiltin<"__builtin_altivec_vaddsbs">,
              Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vadduhs : GCCBuiltin<"__builtin_altivec_vadduhs">,
              Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vaddshs : GCCBuiltin<"__builtin_altivec_vaddshs">,
              Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vadduws : GCCBuiltin<"__builtin_altivec_vadduws">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vaddsws : GCCBuiltin<"__builtin_altivec_vaddsws">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;

  def int_ppc_altivec_vsububs : GCCBuiltin<"__builtin_altivec_vsububs">,
              Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vsubsbs : GCCBuiltin<"__builtin_altivec_vsubsbs">,
              Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vsubuhs : GCCBuiltin<"__builtin_altivec_vsubuhs">,
              Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vsubshs : GCCBuiltin<"__builtin_altivec_vsubshs">,
              Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vsubuws : GCCBuiltin<"__builtin_altivec_vsubuws">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vsubsws : GCCBuiltin<"__builtin_altivec_vsubsws">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;

                        
  // FP to integer conversion.
  def int_ppc_altivec_vcfsx : GCCBuiltin<"__builtin_altivec_vcfsx">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4i32_ty, llvm_int_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcfux : GCCBuiltin<"__builtin_altivec_vcfux">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4i32_ty, llvm_int_ty],
                        [InstrNoMem]>;

  def int_ppc_altivec_vrsqrtefp : GCCBuiltin<"__builtin_altivec_vrsqrtefp">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty], [InstrNoMem]>;
  
}

