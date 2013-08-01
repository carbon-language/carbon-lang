//===- IntrinsicsAArch64.td - Defines AArch64 intrinsics -----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the AArch64-specific intrinsics.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Advanced SIMD (NEON)

let TargetPrefix = "aarch64" in {  // All intrinsics start with "llvm.aarch64.".

// Vector Absolute Compare (Floating Point)
def int_aarch64_neon_vacgeq : Intrinsic<[llvm_v2i64_ty],
                                    [llvm_v2f64_ty, llvm_v2f64_ty],
                                    [IntrNoMem]>;
def int_aarch64_neon_vacgtq : Intrinsic<[llvm_v2i64_ty],
                                    [llvm_v2f64_ty, llvm_v2f64_ty],
                                    [IntrNoMem]>;

// Vector maxNum (Floating Point)
def int_aarch64_neon_vmaxnm : Neon_2Arg_Intrinsic;

// Vector minNum (Floating Point)
def int_aarch64_neon_vminnm : Neon_2Arg_Intrinsic;

// Vector Pairwise maxNum (Floating Point)
def int_aarch64_neon_vpmaxnm : Neon_2Arg_Intrinsic;

// Vector Pairwise minNum (Floating Point)
def int_aarch64_neon_vpminnm : Neon_2Arg_Intrinsic;

// Vector Multiply Extended (Floating Point)
def int_aarch64_neon_vmulx : Neon_2Arg_Intrinsic;
}
