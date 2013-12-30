; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s


define <8 x i8> @mul8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: mul {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = mul <8 x i8> %A, %B;
	ret <8 x i8> %tmp3
}

define <16 x i8> @mul16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: mul {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = mul <16 x i8> %A, %B;
	ret <16 x i8> %tmp3
}

define <4 x i16> @mul4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = mul <4 x i16> %A, %B;
	ret <4 x i16> %tmp3
}

define <8 x i16> @mul8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = mul <8 x i16> %A, %B;
	ret <8 x i16> %tmp3
}

define <2 x i32> @mul2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = mul <2 x i32> %A, %B;
	ret <2 x i32> %tmp3
}

define <4 x i32> @mul4x32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = mul <4 x i32> %A, %B;
	ret <4 x i32> %tmp3
}

define <1 x i64> @mul1xi64(<1 x i64> %A, <1 x i64> %B) {
;CHECK-LABEL: mul1xi64:
;CHECK: mul x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
  %tmp3 = mul <1 x i64> %A, %B;
  ret <1 x i64> %tmp3
}

define <2 x i64> @mul2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK-LABEL: mul2xi64:
;CHECK: mul x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
;CHECK: mul x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
  %tmp3 = mul <2 x i64> %A, %B;
  ret <2 x i64> %tmp3
}

 define <2 x float> @mul2xfloat(<2 x float> %A, <2 x float> %B) {
;CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = fmul <2 x float> %A, %B;
	ret <2 x float> %tmp3
}

define <4 x float> @mul4xfloat(<4 x float> %A, <4 x float> %B) {
;CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = fmul <4 x float> %A, %B;
	ret <4 x float> %tmp3
}
define <2 x double> @mul2xdouble(<2 x double> %A, <2 x double> %B) {
;CHECK: fmul {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = fmul <2 x double> %A, %B;
	ret <2 x double> %tmp3
}


 define <2 x float> @div2xfloat(<2 x float> %A, <2 x float> %B) {
;CHECK: fdiv {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = fdiv <2 x float> %A, %B;
	ret <2 x float> %tmp3
}

define <4 x float> @div4xfloat(<4 x float> %A, <4 x float> %B) {
;CHECK: fdiv {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = fdiv <4 x float> %A, %B;
	ret <4 x float> %tmp3
}
define <2 x double> @div2xdouble(<2 x double> %A, <2 x double> %B) {
;CHECK: fdiv {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = fdiv <2 x double> %A, %B;
	ret <2 x double> %tmp3
}

declare <8 x i8> @llvm.arm.neon.vmulp.v8i8(<8 x i8>, <8 x i8>)
declare <16 x i8> @llvm.arm.neon.vmulp.v16i8(<16 x i8>, <16 x i8>)

define <8 x i8> @poly_mulv8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
; CHECK: poly_mulv8i8:
   %prod = call <8 x i8> @llvm.arm.neon.vmulp.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: pmul v0.8b, v0.8b, v1.8b
   ret <8 x i8> %prod
}

define <16 x i8> @poly_mulv16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: poly_mulv16i8:
   %prod = call <16 x i8> @llvm.arm.neon.vmulp.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: pmul v0.16b, v0.16b, v1.16b
   ret <16 x i8> %prod
}

declare <4 x i16> @llvm.arm.neon.vqdmulh.v4i16(<4 x i16>, <4 x i16>)
declare <8 x i16> @llvm.arm.neon.vqdmulh.v8i16(<8 x i16>, <8 x i16>)
declare <2 x i32> @llvm.arm.neon.vqdmulh.v2i32(<2 x i32>, <2 x i32>)
declare <4 x i32> @llvm.arm.neon.vqdmulh.v4i32(<4 x i32>, <4 x i32>)

define <4 x i16> @test_sqdmulh_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_sqdmulh_v4i16:
   %prod = call <4 x i16> @llvm.arm.neon.vqdmulh.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: sqdmulh v0.4h, v0.4h, v1.4h
   ret <4 x i16> %prod
}

define <8 x i16> @test_sqdmulh_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_sqdmulh_v8i16:
   %prod = call <8 x i16> @llvm.arm.neon.vqdmulh.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: sqdmulh v0.8h, v0.8h, v1.8h
   ret <8 x i16> %prod
}

define <2 x i32> @test_sqdmulh_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_sqdmulh_v2i32:
   %prod = call <2 x i32> @llvm.arm.neon.vqdmulh.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: sqdmulh v0.2s, v0.2s, v1.2s
   ret <2 x i32> %prod
}

define <4 x i32> @test_sqdmulh_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_sqdmulh_v4i32:
   %prod = call <4 x i32> @llvm.arm.neon.vqdmulh.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: sqdmulh v0.4s, v0.4s, v1.4s
   ret <4 x i32> %prod
}

declare <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16>, <4 x i16>)
declare <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16>, <8 x i16>)
declare <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32>, <2 x i32>)
declare <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32>, <4 x i32>)

define <4 x i16> @test_sqrdmulh_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_sqrdmulh_v4i16:
   %prod = call <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: sqrdmulh v0.4h, v0.4h, v1.4h
   ret <4 x i16> %prod
}

define <8 x i16> @test_sqrdmulh_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_sqrdmulh_v8i16:
   %prod = call <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: sqrdmulh v0.8h, v0.8h, v1.8h
   ret <8 x i16> %prod
}

define <2 x i32> @test_sqrdmulh_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_sqrdmulh_v2i32:
   %prod = call <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: sqrdmulh v0.2s, v0.2s, v1.2s
   ret <2 x i32> %prod
}

define <4 x i32> @test_sqrdmulh_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_sqrdmulh_v4i32:
   %prod = call <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: sqrdmulh v0.4s, v0.4s, v1.4s
   ret <4 x i32> %prod
}

declare <2 x float> @llvm.aarch64.neon.vmulx.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.aarch64.neon.vmulx.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.aarch64.neon.vmulx.v2f64(<2 x double>, <2 x double>)

define <2 x float> @fmulx_v2f32(<2 x float> %lhs, <2 x float> %rhs) {
; Using registers other than v0, v1 and v2 are possible, but would be odd.
; CHECK: fmulx v0.2s, v0.2s, v1.2s
        %val = call <2 x float> @llvm.aarch64.neon.vmulx.v2f32(<2 x float> %lhs, <2 x float> %rhs)
        ret <2 x float> %val
}

define <4 x float> @fmulx_v4f32(<4 x float> %lhs, <4 x float> %rhs) {
; Using registers other than v0, v1 and v2 are possible, but would be odd.
; CHECK: fmulx v0.4s, v0.4s, v1.4s
        %val = call <4 x float> @llvm.aarch64.neon.vmulx.v4f32(<4 x float> %lhs, <4 x float> %rhs)
        ret <4 x float> %val
}

define <2 x double> @fmulx_v2f64(<2 x double> %lhs, <2 x double> %rhs) {
; Using registers other than v0, v1 and v2 are possible, but would be odd.
; CHECK: fmulx v0.2d, v0.2d, v1.2d
        %val = call <2 x double> @llvm.aarch64.neon.vmulx.v2f64(<2 x double> %lhs, <2 x double> %rhs)
        ret <2 x double> %val
}
