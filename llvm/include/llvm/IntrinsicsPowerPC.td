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
  // Loads.  These don't map directly to GCC builtins because they represent the
  // source address with a single pointer.
  def int_ppc_altivec_lvx :
              Intrinsic<[llvm_v4i32_ty, llvm_ptr_ty], [IntrReadMem]>;
  def int_ppc_altivec_lvxl :
              Intrinsic<[llvm_v4i32_ty, llvm_ptr_ty], [IntrReadMem]>;
  def int_ppc_altivec_lvebx :
              Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty], [IntrReadMem]>;
  def int_ppc_altivec_lvehx :
              Intrinsic<[llvm_v8i16_ty, llvm_ptr_ty], [IntrReadMem]>;
  def int_ppc_altivec_lvewx :
              Intrinsic<[llvm_v4i32_ty, llvm_ptr_ty], [IntrReadMem]>;

  // Stores.  These don't map directly to GCC builtins because they represent the
  // source address with a single pointer.
  def int_ppc_altivec_stvx :
              Intrinsic<[llvm_void_ty, llvm_v4i32_ty, llvm_ptr_ty],
                        [IntrWriteMem]>;
  def int_ppc_altivec_stvxl :
              Intrinsic<[llvm_void_ty, llvm_v4i32_ty, llvm_ptr_ty],
                        [IntrWriteMem]>;
  def int_ppc_altivec_stvebx :
              Intrinsic<[llvm_void_ty, llvm_v16i8_ty, llvm_ptr_ty],
                        [IntrWriteMem]>;
  def int_ppc_altivec_stvehx :
              Intrinsic<[llvm_void_ty, llvm_v8i16_ty, llvm_ptr_ty],
                        [IntrWriteMem]>;
  def int_ppc_altivec_stvewx :
              Intrinsic<[llvm_void_ty, llvm_v4i32_ty, llvm_ptr_ty],
                        [IntrWriteMem]>;

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

  // Predicate Comparisons.  The first operand specifies interpretation of CR6.
  def int_ppc_altivec_vcmpbfp_p : GCCBuiltin<"__builtin_altivec_vcmpbfp_p">,
              Intrinsic<[llvm_int_ty, llvm_int_ty, llvm_v4f32_ty,llvm_v4f32_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpeqfp_p : GCCBuiltin<"__builtin_altivec_vcmpeqfp_p">,
              Intrinsic<[llvm_int_ty, llvm_int_ty, llvm_v4f32_ty,llvm_v4f32_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpgefp_p : GCCBuiltin<"__builtin_altivec_vcmpgefp_p">,
              Intrinsic<[llvm_int_ty, llvm_int_ty, llvm_v4f32_ty,llvm_v4f32_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpgtfp_p : GCCBuiltin<"__builtin_altivec_vcmpgtfp_p">,
              Intrinsic<[llvm_int_ty, llvm_int_ty, llvm_v4f32_ty,llvm_v4f32_ty],
                        [InstrNoMem]>;
                        
  def int_ppc_altivec_vcmpequw_p : GCCBuiltin<"__builtin_altivec_vcmpequw_p">,
              Intrinsic<[llvm_int_ty, llvm_int_ty, llvm_v4i32_ty,llvm_v4i32_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpgtsw_p : GCCBuiltin<"__builtin_altivec_vcmpgtsw_p">,
              Intrinsic<[llvm_int_ty, llvm_int_ty, llvm_v4i32_ty,llvm_v4i32_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpgtuw_p : GCCBuiltin<"__builtin_altivec_vcmpgtuw_p">,
              Intrinsic<[llvm_int_ty, llvm_int_ty, llvm_v4i32_ty,llvm_v4i32_ty],
                        [InstrNoMem]>;
                        
  def int_ppc_altivec_vcmpequh_p : GCCBuiltin<"__builtin_altivec_vcmpequh_p">,
              Intrinsic<[llvm_int_ty, llvm_int_ty, llvm_v8i16_ty,llvm_v8i16_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpgtsh_p : GCCBuiltin<"__builtin_altivec_vcmpgtsh_p">,
              Intrinsic<[llvm_int_ty, llvm_int_ty, llvm_v8i16_ty,llvm_v8i16_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpgtuh_p : GCCBuiltin<"__builtin_altivec_vcmpgtuh_p">,
              Intrinsic<[llvm_int_ty, llvm_int_ty, llvm_v8i16_ty,llvm_v8i16_ty],
                        [InstrNoMem]>;

  def int_ppc_altivec_vcmpequb_p : GCCBuiltin<"__builtin_altivec_vcmpequb_p">,
              Intrinsic<[llvm_int_ty, llvm_int_ty, llvm_v16i8_ty,llvm_v16i8_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpgtsb_p : GCCBuiltin<"__builtin_altivec_vcmpgtsb_p">,
              Intrinsic<[llvm_int_ty, llvm_int_ty, llvm_v16i8_ty,llvm_v16i8_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcmpgtub_p : GCCBuiltin<"__builtin_altivec_vcmpgtub_p">,
              Intrinsic<[llvm_int_ty, llvm_int_ty, llvm_v16i8_ty,llvm_v16i8_ty],
                        [InstrNoMem]>;

  // Saturating adds, subs, and multiply-adds
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
  def int_ppc_altivec_vmhaddshs : GCCBuiltin<"__builtin_altivec_vmhaddshs">,
              Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty,
                         llvm_v8i16_ty, llvm_v8i16_ty], [InstrNoMem]>;
  def int_ppc_altivec_vmhraddshs : GCCBuiltin<"__builtin_altivec_vmhraddshs">,
              Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty,
                         llvm_v8i16_ty, llvm_v8i16_ty], [InstrNoMem]>;

  def int_ppc_altivec_vmaddfp : GCCBuiltin<"__builtin_altivec_vmaddfp">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_v4f32_ty], [InstrNoMem]>;
  def int_ppc_altivec_vaddcuw : GCCBuiltin<"__builtin_altivec_vaddcuw">,
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

  def int_ppc_altivec_vnmsubfp : GCCBuiltin<"__builtin_altivec_vnmsubfp">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_v4f32_ty], [InstrNoMem]>;
  def int_ppc_altivec_vsubcuw : GCCBuiltin<"__builtin_altivec_vsubcuw">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;

  // Vector Multiply Sum Instructions.
  def int_ppc_altivec_vmsummbm : GCCBuiltin<"__builtin_altivec_vmsummbm">,
            Intrinsic<[llvm_v4i32_ty, llvm_v16i8_ty, llvm_v16i8_ty,
                       llvm_v4i32_ty], [InstrNoMem]>;
  def int_ppc_altivec_vmsumshm : GCCBuiltin<"__builtin_altivec_vmsumshm">,
            Intrinsic<[llvm_v4i32_ty, llvm_v8i16_ty, llvm_v8i16_ty,
                       llvm_v4i32_ty], [InstrNoMem]>;
  def int_ppc_altivec_vmsumshs : GCCBuiltin<"__builtin_altivec_vmsumshs">,
            Intrinsic<[llvm_v4i32_ty, llvm_v8i16_ty, llvm_v8i16_ty, 
                       llvm_v4i32_ty], [InstrNoMem]>;
  def int_ppc_altivec_vmsumubm : GCCBuiltin<"__builtin_altivec_vmsumubm">,
            Intrinsic<[llvm_v4i32_ty, llvm_v16i8_ty, llvm_v16i8_ty, 
                       llvm_v4i32_ty], [InstrNoMem]>;
  def int_ppc_altivec_vmsumuhm : GCCBuiltin<"__builtin_altivec_vmsumuhm">,
            Intrinsic<[llvm_v4i32_ty, llvm_v8i16_ty, llvm_v8i16_ty,
                       llvm_v4i32_ty], [InstrNoMem]>;
  def int_ppc_altivec_vmsumuhs : GCCBuiltin<"__builtin_altivec_vmsumuhs">,
            Intrinsic<[llvm_v4i32_ty, llvm_v8i16_ty, llvm_v8i16_ty,
                       llvm_v4i32_ty], [InstrNoMem]>;

  // Vector Multiply Instructions.
  def int_ppc_altivec_vmulesb : GCCBuiltin<"__builtin_altivec_vmulesb">,
          Intrinsic<[llvm_v8i16_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                    [InstrNoMem]>;
  def int_ppc_altivec_vmulesh : GCCBuiltin<"__builtin_altivec_vmulesh">,
          Intrinsic<[llvm_v4i32_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                    [InstrNoMem]>;
  def int_ppc_altivec_vmuleub : GCCBuiltin<"__builtin_altivec_vmuleub">,
          Intrinsic<[llvm_v8i16_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                    [InstrNoMem]>;
  def int_ppc_altivec_vmuleuh : GCCBuiltin<"__builtin_altivec_vmuleuh">,
          Intrinsic<[llvm_v4i32_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                    [InstrNoMem]>;

  def int_ppc_altivec_vmulosb : GCCBuiltin<"__builtin_altivec_vmulosb">,
          Intrinsic<[llvm_v8i16_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                    [InstrNoMem]>;
  def int_ppc_altivec_vmulosh : GCCBuiltin<"__builtin_altivec_vmulosh">,
          Intrinsic<[llvm_v4i32_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                    [InstrNoMem]>;
  def int_ppc_altivec_vmuloub : GCCBuiltin<"__builtin_altivec_vmuloub">,
          Intrinsic<[llvm_v8i16_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                    [InstrNoMem]>;
  def int_ppc_altivec_vmulouh : GCCBuiltin<"__builtin_altivec_vmulouh">,
          Intrinsic<[llvm_v4i32_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                    [InstrNoMem]>;

  // Vector Sum Instructions.
  def int_ppc_altivec_vsumsws : GCCBuiltin<"__builtin_altivec_vsumsws">,
            Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                      [InstrNoMem]>;
  def int_ppc_altivec_vsum2sws : GCCBuiltin<"__builtin_altivec_vsum2sws">,
            Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                      [InstrNoMem]>;
  def int_ppc_altivec_vsum4sbs : GCCBuiltin<"__builtin_altivec_vsum4sbs">,
            Intrinsic<[llvm_v4i32_ty, llvm_v16i8_ty, llvm_v4i32_ty],
                      [InstrNoMem]>;
  def int_ppc_altivec_vsum4shs : GCCBuiltin<"__builtin_altivec_vsum4shs">,
            Intrinsic<[llvm_v4i32_ty, llvm_v8i16_ty, llvm_v4i32_ty],
                      [InstrNoMem]>;
  def int_ppc_altivec_vsum4ubs : GCCBuiltin<"__builtin_altivec_vsum4ubs">,
            Intrinsic<[llvm_v4i32_ty, llvm_v16i8_ty, llvm_v4i32_ty],
                      [InstrNoMem]>;

  // Packs.
  def int_ppc_altivec_vpkpx : GCCBuiltin<"__builtin_altivec_vpkpx">,
            Intrinsic<[llvm_v8i16_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                      [InstrNoMem]>;
  def int_ppc_altivec_vpkshss : GCCBuiltin<"__builtin_altivec_vpkshss">,
            Intrinsic<[llvm_v16i8_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                      [InstrNoMem]>;
  def int_ppc_altivec_vpkshus : GCCBuiltin<"__builtin_altivec_vpkshus">,
            Intrinsic<[llvm_v16i8_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                      [InstrNoMem]>;
  def int_ppc_altivec_vpkswss : GCCBuiltin<"__builtin_altivec_vpkswss">,
            Intrinsic<[llvm_v4i32_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                      [InstrNoMem]>;
  def int_ppc_altivec_vpkswus : GCCBuiltin<"__builtin_altivec_vpkswus">,
            Intrinsic<[llvm_v4i32_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                      [InstrNoMem]>;
  // vpkuhum is lowered to a shuffle.
  def int_ppc_altivec_vpkuhus : GCCBuiltin<"__builtin_altivec_vpkuhus">,
            Intrinsic<[llvm_v16i8_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                      [InstrNoMem]>;
  // vpkuwum is lowered to a shuffle.
  def int_ppc_altivec_vpkuwus : GCCBuiltin<"__builtin_altivec_vpkuwus">,
            Intrinsic<[llvm_v8i16_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                      [InstrNoMem]>;

  // Unpacks.
  def int_ppc_altivec_vupkhpx : GCCBuiltin<"__builtin_altivec_vupkhpx">,
            Intrinsic<[llvm_v4i32_ty, llvm_v8i16_ty], [InstrNoMem]>;
  def int_ppc_altivec_vupkhsb : GCCBuiltin<"__builtin_altivec_vupkhsb">,
            Intrinsic<[llvm_v8i16_ty, llvm_v16i8_ty], [InstrNoMem]>;
  def int_ppc_altivec_vupkhsh : GCCBuiltin<"__builtin_altivec_vupkhsh">,
            Intrinsic<[llvm_v4i32_ty, llvm_v8i16_ty], [InstrNoMem]>;
  def int_ppc_altivec_vupklpx : GCCBuiltin<"__builtin_altivec_vupklpx">,
            Intrinsic<[llvm_v4i32_ty, llvm_v8i16_ty], [InstrNoMem]>;
  def int_ppc_altivec_vupklsb : GCCBuiltin<"__builtin_altivec_vupklsb">,
            Intrinsic<[llvm_v8i16_ty, llvm_v16i8_ty], [InstrNoMem]>;
  def int_ppc_altivec_vupklsh : GCCBuiltin<"__builtin_altivec_vupklsh">,
            Intrinsic<[llvm_v4i32_ty, llvm_v8i16_ty], [InstrNoMem]>;


  // FP to integer conversion.
  def int_ppc_altivec_vcfsx : GCCBuiltin<"__builtin_altivec_vcfsx">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4i32_ty, llvm_int_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vcfux : GCCBuiltin<"__builtin_altivec_vcfux">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4i32_ty, llvm_int_ty],
                        [InstrNoMem]>;

  def int_ppc_altivec_vrfim : GCCBuiltin<"__builtin_altivec_vrfim">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty], [InstrNoMem]>;
  def int_ppc_altivec_vrfin : GCCBuiltin<"__builtin_altivec_vrfin">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty], [InstrNoMem]>;
  def int_ppc_altivec_vrfip : GCCBuiltin<"__builtin_altivec_vrfip">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty], [InstrNoMem]>;
  def int_ppc_altivec_vrfiz : GCCBuiltin<"__builtin_altivec_vrfiz">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty], [InstrNoMem]>;

  // Merges
  def int_ppc_altivec_vmrghh : GCCBuiltin<"__builtin_altivec_vmrghh">,
              Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vmrghw : GCCBuiltin<"__builtin_altivec_vmrghw">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vmrglh : GCCBuiltin<"__builtin_altivec_vmrglh">,
              Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vmrglw : GCCBuiltin<"__builtin_altivec_vmrglw">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;

  // Left Shifts.
  def int_ppc_altivec_vsldoi : GCCBuiltin<"__builtin_altivec_vsldoi_4si">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, 
                         llvm_v4i32_ty, llvm_int_ty], [InstrNoMem]>;
  def int_ppc_altivec_vslo : GCCBuiltin<"__builtin_altivec_vslo">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;

  def int_ppc_altivec_vslb : GCCBuiltin<"__builtin_altivec_vslb">,
              Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vslh : GCCBuiltin<"__builtin_altivec_vslh">,
              Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vslw : GCCBuiltin<"__builtin_altivec_vslw">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;

  // Right Shifts.
  def int_ppc_altivec_vsr : GCCBuiltin<"__builtin_altivec_vsr">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vsro : GCCBuiltin<"__builtin_altivec_vsro">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;

  def int_ppc_altivec_vsrb : GCCBuiltin<"__builtin_altivec_vsrb">,
              Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vsrh : GCCBuiltin<"__builtin_altivec_vsrh">,
              Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vsrw : GCCBuiltin<"__builtin_altivec_vsrw">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vsrab : GCCBuiltin<"__builtin_altivec_vsrab">,
              Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vsrah : GCCBuiltin<"__builtin_altivec_vsrah">,
              Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vsraw : GCCBuiltin<"__builtin_altivec_vsraw">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;

  // Rotates.
  def int_ppc_altivec_vrlb : GCCBuiltin<"__builtin_altivec_vrlb">,
              Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vrlh : GCCBuiltin<"__builtin_altivec_vrlh">,
              Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                        [InstrNoMem]>;
  def int_ppc_altivec_vrlw : GCCBuiltin<"__builtin_altivec_vrlw">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;

  // Miscellaneous.
  def int_ppc_altivec_lvsl :
              Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty], [InstrNoMem]>;
  def int_ppc_altivec_lvsr :
              Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty], [InstrNoMem]>;

  def int_ppc_altivec_vperm : GCCBuiltin<"__builtin_altivec_vperm_4si">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, 
                         llvm_v4i32_ty, llvm_v16i8_ty], [InstrNoMem]>;
  def int_ppc_altivec_vsel : GCCBuiltin<"__builtin_altivec_vsel_4si">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, 
                         llvm_v4i32_ty, llvm_v4i32_ty], [InstrNoMem]>;

  def int_ppc_altivec_vexptefp : GCCBuiltin<"__builtin_altivec_vexptefp">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty], [InstrNoMem]>;
  def int_ppc_altivec_vlogefp : GCCBuiltin<"__builtin_altivec_vlogefp">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty], [InstrNoMem]>;
  def int_ppc_altivec_vrefp : GCCBuiltin<"__builtin_altivec_vrefp">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty], [InstrNoMem]>;
  def int_ppc_altivec_vrsqrtefp : GCCBuiltin<"__builtin_altivec_vrsqrtefp">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty], [InstrNoMem]>;
}

