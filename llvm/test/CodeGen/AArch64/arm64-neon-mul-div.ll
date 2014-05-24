; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-none-linux-gnu -mattr=+neon | FileCheck %s
; arm64 has its own copy of this because of the intrinsics

define <8 x i8> @mul8xi8(<8 x i8> %A, <8 x i8> %B) {
; CHECK-LABEL: mul8xi8:
; CHECK: mul {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = mul <8 x i8> %A, %B;
	ret <8 x i8> %tmp3
}

define <16 x i8> @mul16xi8(<16 x i8> %A, <16 x i8> %B) {
; CHECK-LABEL: mul16xi8:
; CHECK: mul {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = mul <16 x i8> %A, %B;
	ret <16 x i8> %tmp3
}

define <4 x i16> @mul4xi16(<4 x i16> %A, <4 x i16> %B) {
; CHECK-LABEL: mul4xi16:
; CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = mul <4 x i16> %A, %B;
	ret <4 x i16> %tmp3
}

define <8 x i16> @mul8xi16(<8 x i16> %A, <8 x i16> %B) {
; CHECK-LABEL: mul8xi16:
; CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = mul <8 x i16> %A, %B;
	ret <8 x i16> %tmp3
}

define <2 x i32> @mul2xi32(<2 x i32> %A, <2 x i32> %B) {
; CHECK-LABEL: mul2xi32:
; CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = mul <2 x i32> %A, %B;
	ret <2 x i32> %tmp3
}

define <4 x i32> @mul4x32(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: mul4x32:
; CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = mul <4 x i32> %A, %B;
	ret <4 x i32> %tmp3
}

define <1 x i64> @mul1xi64(<1 x i64> %A, <1 x i64> %B) {
; CHECK-LABEL: mul1xi64:
; CHECK: mul x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
  %tmp3 = mul <1 x i64> %A, %B;
  ret <1 x i64> %tmp3
}

define <2 x i64> @mul2xi64(<2 x i64> %A, <2 x i64> %B) {
; CHECK-LABEL: mul2xi64:
; CHECK: mul x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
; CHECK: mul x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
  %tmp3 = mul <2 x i64> %A, %B;
  ret <2 x i64> %tmp3
}

 define <2 x float> @mul2xfloat(<2 x float> %A, <2 x float> %B) {
; CHECK-LABEL: mul2xfloat:
; CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = fmul <2 x float> %A, %B;
	ret <2 x float> %tmp3
}

define <4 x float> @mul4xfloat(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: mul4xfloat:
; CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = fmul <4 x float> %A, %B;
	ret <4 x float> %tmp3
}
define <2 x double> @mul2xdouble(<2 x double> %A, <2 x double> %B) {
; CHECK-LABEL: mul2xdouble:
; CHECK: fmul {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = fmul <2 x double> %A, %B;
	ret <2 x double> %tmp3
}


 define <2 x float> @div2xfloat(<2 x float> %A, <2 x float> %B) {
; CHECK-LABEL: div2xfloat:
; CHECK: fdiv {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = fdiv <2 x float> %A, %B;
	ret <2 x float> %tmp3
}

define <4 x float> @div4xfloat(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: div4xfloat:
; CHECK: fdiv {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = fdiv <4 x float> %A, %B;
	ret <4 x float> %tmp3
}
define <2 x double> @div2xdouble(<2 x double> %A, <2 x double> %B) {
; CHECK-LABEL: div2xdouble:
; CHECK: fdiv {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = fdiv <2 x double> %A, %B;
	ret <2 x double> %tmp3
}

define <1 x i8> @sdiv1x8(<1 x i8> %A, <1 x i8> %B) {
; CHECK-LABEL: sdiv1x8:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = sdiv <1 x i8> %A, %B;
	ret <1 x i8> %tmp3
}

define <8 x i8> @sdiv8x8(<8 x i8> %A, <8 x i8> %B) {
; CHECK-LABEL: sdiv8x8:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = sdiv <8 x i8> %A, %B;
	ret <8 x i8> %tmp3
}

define <16 x i8> @sdiv16x8(<16 x i8> %A, <16 x i8> %B) {
; CHECK-LABEL: sdiv16x8:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = sdiv <16 x i8> %A, %B;
	ret <16 x i8> %tmp3
}

define <1 x i16> @sdiv1x16(<1 x i16> %A, <1 x i16> %B) {
; CHECK-LABEL: sdiv1x16:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = sdiv <1 x i16> %A, %B;
	ret <1 x i16> %tmp3
}

define <4 x i16> @sdiv4x16(<4 x i16> %A, <4 x i16> %B) {
; CHECK-LABEL: sdiv4x16:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = sdiv <4 x i16> %A, %B;
	ret <4 x i16> %tmp3
}

define <8 x i16> @sdiv8x16(<8 x i16> %A, <8 x i16> %B) {
; CHECK-LABEL: sdiv8x16:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = sdiv <8 x i16> %A, %B;
	ret <8 x i16> %tmp3
}

define <1 x i32> @sdiv1x32(<1 x i32> %A, <1 x i32> %B) {
; CHECK-LABEL: sdiv1x32:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = sdiv <1 x i32> %A, %B;
	ret <1 x i32> %tmp3
}

define <2 x i32> @sdiv2x32(<2 x i32> %A, <2 x i32> %B) {
; CHECK-LABEL: sdiv2x32:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = sdiv <2 x i32> %A, %B;
	ret <2 x i32> %tmp3
}

define <4 x i32> @sdiv4x32(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: sdiv4x32:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = sdiv <4 x i32> %A, %B;
	ret <4 x i32> %tmp3
}

define <1 x i64> @sdiv1x64(<1 x i64> %A, <1 x i64> %B) {
; CHECK-LABEL: sdiv1x64:
; CHECK: sdiv {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
	%tmp3 = sdiv <1 x i64> %A, %B;
	ret <1 x i64> %tmp3
}

define <2 x i64> @sdiv2x64(<2 x i64> %A, <2 x i64> %B) {
; CHECK-LABEL: sdiv2x64:
; CHECK: sdiv {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK: sdiv {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
	%tmp3 = sdiv <2 x i64> %A, %B;
	ret <2 x i64> %tmp3
}

define <1 x i8> @udiv1x8(<1 x i8> %A, <1 x i8> %B) {
; CHECK-LABEL: udiv1x8:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = udiv <1 x i8> %A, %B;
	ret <1 x i8> %tmp3
}

define <8 x i8> @udiv8x8(<8 x i8> %A, <8 x i8> %B) {
; CHECK-LABEL: udiv8x8:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = udiv <8 x i8> %A, %B;
	ret <8 x i8> %tmp3
}

define <16 x i8> @udiv16x8(<16 x i8> %A, <16 x i8> %B) {
; CHECK-LABEL: udiv16x8:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = udiv <16 x i8> %A, %B;
	ret <16 x i8> %tmp3
}

define <1 x i16> @udiv1x16(<1 x i16> %A, <1 x i16> %B) {
; CHECK-LABEL: udiv1x16:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = udiv <1 x i16> %A, %B;
	ret <1 x i16> %tmp3
}

define <4 x i16> @udiv4x16(<4 x i16> %A, <4 x i16> %B) {
; CHECK-LABEL: udiv4x16:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = udiv <4 x i16> %A, %B;
	ret <4 x i16> %tmp3
}

define <8 x i16> @udiv8x16(<8 x i16> %A, <8 x i16> %B) {
; CHECK-LABEL: udiv8x16:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = udiv <8 x i16> %A, %B;
	ret <8 x i16> %tmp3
}

define <1 x i32> @udiv1x32(<1 x i32> %A, <1 x i32> %B) {
; CHECK-LABEL: udiv1x32:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = udiv <1 x i32> %A, %B;
	ret <1 x i32> %tmp3
}

define <2 x i32> @udiv2x32(<2 x i32> %A, <2 x i32> %B) {
; CHECK-LABEL: udiv2x32:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = udiv <2 x i32> %A, %B;
	ret <2 x i32> %tmp3
}

define <4 x i32> @udiv4x32(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: udiv4x32:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = udiv <4 x i32> %A, %B;
	ret <4 x i32> %tmp3
}

define <1 x i64> @udiv1x64(<1 x i64> %A, <1 x i64> %B) {
; CHECK-LABEL: udiv1x64:
; CHECK: udiv {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
	%tmp3 = udiv <1 x i64> %A, %B;
	ret <1 x i64> %tmp3
}

define <2 x i64> @udiv2x64(<2 x i64> %A, <2 x i64> %B) {
; CHECK-LABEL: udiv2x64:
; CHECK: udiv {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK: udiv {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
	%tmp3 = udiv <2 x i64> %A, %B;
	ret <2 x i64> %tmp3
}

define <1 x i8> @srem1x8(<1 x i8> %A, <1 x i8> %B) {
; CHECK-LABEL: srem1x8:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = srem <1 x i8> %A, %B;
	ret <1 x i8> %tmp3
}

define <8 x i8> @srem8x8(<8 x i8> %A, <8 x i8> %B) {
; CHECK-LABEL: srem8x8:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = srem <8 x i8> %A, %B;
	ret <8 x i8> %tmp3
}

define <16 x i8> @srem16x8(<16 x i8> %A, <16 x i8> %B) {
; CHECK-LABEL: srem16x8:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = srem <16 x i8> %A, %B;
	ret <16 x i8> %tmp3
}

define <1 x i16> @srem1x16(<1 x i16> %A, <1 x i16> %B) {
; CHECK-LABEL: srem1x16:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = srem <1 x i16> %A, %B;
	ret <1 x i16> %tmp3
}

define <4 x i16> @srem4x16(<4 x i16> %A, <4 x i16> %B) {
; CHECK-LABEL: srem4x16:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = srem <4 x i16> %A, %B;
	ret <4 x i16> %tmp3
}

define <8 x i16> @srem8x16(<8 x i16> %A, <8 x i16> %B) {
; CHECK-LABEL: srem8x16:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = srem <8 x i16> %A, %B;
	ret <8 x i16> %tmp3
}

define <1 x i32> @srem1x32(<1 x i32> %A, <1 x i32> %B) {
; CHECK-LABEL: srem1x32:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = srem <1 x i32> %A, %B;
	ret <1 x i32> %tmp3
}

define <2 x i32> @srem2x32(<2 x i32> %A, <2 x i32> %B) {
; CHECK-LABEL: srem2x32:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = srem <2 x i32> %A, %B;
	ret <2 x i32> %tmp3
}

define <4 x i32> @srem4x32(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: srem4x32:
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: sdiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = srem <4 x i32> %A, %B;
	ret <4 x i32> %tmp3
}

define <1 x i64> @srem1x64(<1 x i64> %A, <1 x i64> %B) {
; CHECK-LABEL: srem1x64:
; CHECK: sdiv {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK: msub {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
	%tmp3 = srem <1 x i64> %A, %B;
	ret <1 x i64> %tmp3
}

define <2 x i64> @srem2x64(<2 x i64> %A, <2 x i64> %B) {
; CHECK-LABEL: srem2x64:
; CHECK: sdiv {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK: sdiv {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK: msub {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK: msub {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
	%tmp3 = srem <2 x i64> %A, %B;
	ret <2 x i64> %tmp3
}

define <1 x i8> @urem1x8(<1 x i8> %A, <1 x i8> %B) {
; CHECK-LABEL: urem1x8:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = urem <1 x i8> %A, %B;
	ret <1 x i8> %tmp3
}

define <8 x i8> @urem8x8(<8 x i8> %A, <8 x i8> %B) {
; CHECK-LABEL: urem8x8:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = urem <8 x i8> %A, %B;
	ret <8 x i8> %tmp3
}

define <16 x i8> @urem16x8(<16 x i8> %A, <16 x i8> %B) {
; CHECK-LABEL: urem16x8:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = urem <16 x i8> %A, %B;
	ret <16 x i8> %tmp3
}

define <1 x i16> @urem1x16(<1 x i16> %A, <1 x i16> %B) {
; CHECK-LABEL: urem1x16:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = urem <1 x i16> %A, %B;
	ret <1 x i16> %tmp3
}

define <4 x i16> @urem4x16(<4 x i16> %A, <4 x i16> %B) {
; CHECK-LABEL: urem4x16:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = urem <4 x i16> %A, %B;
	ret <4 x i16> %tmp3
}

define <8 x i16> @urem8x16(<8 x i16> %A, <8 x i16> %B) {
; CHECK-LABEL: urem8x16:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = urem <8 x i16> %A, %B;
	ret <8 x i16> %tmp3
}

define <1 x i32> @urem1x32(<1 x i32> %A, <1 x i32> %B) {
; CHECK-LABEL: urem1x32:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = urem <1 x i32> %A, %B;
	ret <1 x i32> %tmp3
}

define <2 x i32> @urem2x32(<2 x i32> %A, <2 x i32> %B) {
; CHECK-LABEL: urem2x32:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = urem <2 x i32> %A, %B;
	ret <2 x i32> %tmp3
}

define <4 x i32> @urem4x32(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: urem4x32:
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: udiv {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: msub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
	%tmp3 = urem <4 x i32> %A, %B;
	ret <4 x i32> %tmp3
}

define <1 x i64> @urem1x64(<1 x i64> %A, <1 x i64> %B) {
; CHECK-LABEL: urem1x64:
; CHECK: udiv {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK: msub {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
	%tmp3 = urem <1 x i64> %A, %B;
	ret <1 x i64> %tmp3
}

define <2 x i64> @urem2x64(<2 x i64> %A, <2 x i64> %B) {
; CHECK-LABEL: urem2x64:
; CHECK: udiv {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK: udiv {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK: msub {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK: msub {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
	%tmp3 = urem <2 x i64> %A, %B;
	ret <2 x i64> %tmp3
}

define <2 x float> @frem2f32(<2 x float> %A, <2 x float> %B) {
; CHECK-LABEL: frem2f32:
; CHECK: bl fmodf
; CHECK: bl fmodf
	%tmp3 = frem <2 x float> %A, %B;
	ret <2 x float> %tmp3
}

define <4 x float> @frem4f32(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: frem4f32:
; CHECK: bl fmodf
; CHECK: bl fmodf
; CHECK: bl fmodf
; CHECK: bl fmodf
	%tmp3 = frem <4 x float> %A, %B;
	ret <4 x float> %tmp3
}

define <1 x double> @frem1d64(<1 x double> %A, <1 x double> %B) {
; CHECK-LABEL: frem1d64:
; CHECK: bl fmod
	%tmp3 = frem <1 x double> %A, %B;
	ret <1 x double> %tmp3
}

define <2 x double> @frem2d64(<2 x double> %A, <2 x double> %B) {
; CHECK-LABEL: frem2d64:
; CHECK: bl fmod
; CHECK: bl fmod
	%tmp3 = frem <2 x double> %A, %B;
	ret <2 x double> %tmp3
}

declare <8 x i8> @llvm.aarch64.neon.pmul.v8i8(<8 x i8>, <8 x i8>)
declare <16 x i8> @llvm.aarch64.neon.pmul.v16i8(<16 x i8>, <16 x i8>)

define <8 x i8> @poly_mulv8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
; CHECK-LABEL: poly_mulv8i8:
   %prod = call <8 x i8> @llvm.aarch64.neon.pmul.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: pmul v0.8b, v0.8b, v1.8b
   ret <8 x i8> %prod
}

define <16 x i8> @poly_mulv16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK-LABEL: poly_mulv16i8:
   %prod = call <16 x i8> @llvm.aarch64.neon.pmul.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: pmul v0.16b, v0.16b, v1.16b
   ret <16 x i8> %prod
}

declare <4 x i16> @llvm.aarch64.neon.sqdmulh.v4i16(<4 x i16>, <4 x i16>)
declare <8 x i16> @llvm.aarch64.neon.sqdmulh.v8i16(<8 x i16>, <8 x i16>)
declare <2 x i32> @llvm.aarch64.neon.sqdmulh.v2i32(<2 x i32>, <2 x i32>)
declare <4 x i32> @llvm.aarch64.neon.sqdmulh.v4i32(<4 x i32>, <4 x i32>)

define <4 x i16> @test_sqdmulh_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK-LABEL: test_sqdmulh_v4i16:
   %prod = call <4 x i16> @llvm.aarch64.neon.sqdmulh.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: sqdmulh v0.4h, v0.4h, v1.4h
   ret <4 x i16> %prod
}

define <8 x i16> @test_sqdmulh_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK-LABEL: test_sqdmulh_v8i16:
   %prod = call <8 x i16> @llvm.aarch64.neon.sqdmulh.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: sqdmulh v0.8h, v0.8h, v1.8h
   ret <8 x i16> %prod
}

define <2 x i32> @test_sqdmulh_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK-LABEL: test_sqdmulh_v2i32:
   %prod = call <2 x i32> @llvm.aarch64.neon.sqdmulh.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: sqdmulh v0.2s, v0.2s, v1.2s
   ret <2 x i32> %prod
}

define <4 x i32> @test_sqdmulh_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK-LABEL: test_sqdmulh_v4i32:
   %prod = call <4 x i32> @llvm.aarch64.neon.sqdmulh.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: sqdmulh v0.4s, v0.4s, v1.4s
   ret <4 x i32> %prod
}

declare <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16>, <4 x i16>)
declare <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16>, <8 x i16>)
declare <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32>, <2 x i32>)
declare <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32>, <4 x i32>)

define <4 x i16> @test_sqrdmulh_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK-LABEL: test_sqrdmulh_v4i16:
   %prod = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: sqrdmulh v0.4h, v0.4h, v1.4h
   ret <4 x i16> %prod
}

define <8 x i16> @test_sqrdmulh_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK-LABEL: test_sqrdmulh_v8i16:
   %prod = call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: sqrdmulh v0.8h, v0.8h, v1.8h
   ret <8 x i16> %prod
}

define <2 x i32> @test_sqrdmulh_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK-LABEL: test_sqrdmulh_v2i32:
   %prod = call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: sqrdmulh v0.2s, v0.2s, v1.2s
   ret <2 x i32> %prod
}

define <4 x i32> @test_sqrdmulh_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK-LABEL: test_sqrdmulh_v4i32:
   %prod = call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: sqrdmulh v0.4s, v0.4s, v1.4s
   ret <4 x i32> %prod
}

declare <2 x float> @llvm.aarch64.neon.fmulx.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.aarch64.neon.fmulx.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.aarch64.neon.fmulx.v2f64(<2 x double>, <2 x double>)

define <2 x float> @fmulx_v2f32(<2 x float> %lhs, <2 x float> %rhs) {
; CHECK-LABEL: fmulx_v2f32:
; Using registers other than v0, v1 and v2 are possible, but would be odd.
; CHECK: fmulx v0.2s, v0.2s, v1.2s
        %val = call <2 x float> @llvm.aarch64.neon.fmulx.v2f32(<2 x float> %lhs, <2 x float> %rhs)
        ret <2 x float> %val
}

define <4 x float> @fmulx_v4f32(<4 x float> %lhs, <4 x float> %rhs) {
; CHECK-LABEL: fmulx_v4f32:
; Using registers other than v0, v1 and v2 are possible, but would be odd.
; CHECK: fmulx v0.4s, v0.4s, v1.4s
        %val = call <4 x float> @llvm.aarch64.neon.fmulx.v4f32(<4 x float> %lhs, <4 x float> %rhs)
        ret <4 x float> %val
}

define <2 x double> @fmulx_v2f64(<2 x double> %lhs, <2 x double> %rhs) {
; CHECK-LABEL: fmulx_v2f64:
; Using registers other than v0, v1 and v2 are possible, but would be odd.
; CHECK: fmulx v0.2d, v0.2d, v1.2d
        %val = call <2 x double> @llvm.aarch64.neon.fmulx.v2f64(<2 x double> %lhs, <2 x double> %rhs)
        ret <2 x double> %val
}

