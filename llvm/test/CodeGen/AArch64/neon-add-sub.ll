; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define <8 x i8> @add8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: add {{v[0-31]+}}.8b, {{v[0-31]+}}.8b, {{v[0-31]+}}.8b
	%tmp3 = add <8 x i8> %A, %B;
	ret <8 x i8> %tmp3
}

define <16 x i8> @add16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: add {{v[0-31]+}}.16b, {{v[0-31]+}}.16b, {{v[0-31]+}}.16b
	%tmp3 = add <16 x i8> %A, %B;
	ret <16 x i8> %tmp3
}

define <4 x i16> @add4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: add {{v[0-31]+}}.4h, {{v[0-31]+}}.4h, {{v[0-31]+}}.4h
	%tmp3 = add <4 x i16> %A, %B;
	ret <4 x i16> %tmp3
}

define <8 x i16> @add8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: add {{v[0-31]+}}.8h, {{v[0-31]+}}.8h, {{v[0-31]+}}.8h
	%tmp3 = add <8 x i16> %A, %B;
	ret <8 x i16> %tmp3
}

define <2 x i32> @add2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: add {{v[0-31]+}}.2s, {{v[0-31]+}}.2s, {{v[0-31]+}}.2s
	%tmp3 = add <2 x i32> %A, %B;
	ret <2 x i32> %tmp3
}

define <4 x i32> @add4x32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: add {{v[0-31]+}}.4s, {{v[0-31]+}}.4s, {{v[0-31]+}}.4s
	%tmp3 = add <4 x i32> %A, %B;
	ret <4 x i32> %tmp3
}

define <2 x i64> @add2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK: add {{v[0-31]+}}.2d, {{v[0-31]+}}.2d, {{v[0-31]+}}.2d
	%tmp3 = add <2 x i64> %A, %B;
	ret <2 x i64> %tmp3
}

define <2 x float> @add2xfloat(<2 x float> %A, <2 x float> %B) {
;CHECK: fadd {{v[0-31]+}}.2s, {{v[0-31]+}}.2s, {{v[0-31]+}}.2s
	%tmp3 = fadd <2 x float> %A, %B;
	ret <2 x float> %tmp3
}

define <4 x float> @add4xfloat(<4 x float> %A, <4 x float> %B) {
;CHECK: fadd {{v[0-31]+}}.4s, {{v[0-31]+}}.4s, {{v[0-31]+}}.4s
	%tmp3 = fadd <4 x float> %A, %B;
	ret <4 x float> %tmp3
}
define <2 x double> @add2xdouble(<2 x double> %A, <2 x double> %B) {
;CHECK: add {{v[0-31]+}}.2d, {{v[0-31]+}}.2d, {{v[0-31]+}}.2d
	%tmp3 = fadd <2 x double> %A, %B;
	ret <2 x double> %tmp3
}

define <8 x i8> @sub8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: sub {{v[0-31]+}}.8b, {{v[0-31]+}}.8b, {{v[0-31]+}}.8b
	%tmp3 = sub <8 x i8> %A, %B;
	ret <8 x i8> %tmp3
}

define <16 x i8> @sub16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: sub {{v[0-31]+}}.16b, {{v[0-31]+}}.16b, {{v[0-31]+}}.16b
	%tmp3 = sub <16 x i8> %A, %B;
	ret <16 x i8> %tmp3
}

define <4 x i16> @sub4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: sub {{v[0-31]+}}.4h, {{v[0-31]+}}.4h, {{v[0-31]+}}.4h
	%tmp3 = sub <4 x i16> %A, %B;
	ret <4 x i16> %tmp3
}

define <8 x i16> @sub8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: sub {{v[0-31]+}}.8h, {{v[0-31]+}}.8h, {{v[0-31]+}}.8h
	%tmp3 = sub <8 x i16> %A, %B;
	ret <8 x i16> %tmp3
}

define <2 x i32> @sub2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: sub {{v[0-31]+}}.2s, {{v[0-31]+}}.2s, {{v[0-31]+}}.2s
	%tmp3 = sub <2 x i32> %A, %B;
	ret <2 x i32> %tmp3
}

define <4 x i32> @sub4x32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: sub {{v[0-31]+}}.4s, {{v[0-31]+}}.4s, {{v[0-31]+}}.4s
	%tmp3 = sub <4 x i32> %A, %B;
	ret <4 x i32> %tmp3
}

define <2 x i64> @sub2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK: sub {{v[0-31]+}}.2d, {{v[0-31]+}}.2d, {{v[0-31]+}}.2d
	%tmp3 = sub <2 x i64> %A, %B;
	ret <2 x i64> %tmp3
}

define <2 x float> @sub2xfloat(<2 x float> %A, <2 x float> %B) {
;CHECK: fsub {{v[0-31]+}}.2s, {{v[0-31]+}}.2s, {{v[0-31]+}}.2s
	%tmp3 = fsub <2 x float> %A, %B;
	ret <2 x float> %tmp3
}

define <4 x float> @sub4xfloat(<4 x float> %A, <4 x float> %B) {
;CHECK: fsub {{v[0-31]+}}.4s, {{v[0-31]+}}.4s, {{v[0-31]+}}.4s
	%tmp3 = fsub <4 x float> %A, %B;
	ret <4 x float> %tmp3
}
define <2 x double> @sub2xdouble(<2 x double> %A, <2 x double> %B) {
;CHECK: sub {{v[0-31]+}}.2d, {{v[0-31]+}}.2d, {{v[0-31]+}}.2d
	%tmp3 = fsub <2 x double> %A, %B;
	ret <2 x double> %tmp3
}

define <1 x i64> @add1xi64(<1 x i64> %A, <1 x i64> %B) {
;CHECK: add {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
	%tmp3 = add <1 x i64> %A, %B;
	ret <1 x i64> %tmp3
}

define <1 x i64> @sub1xi64(<1 x i64> %A, <1 x i64> %B) {
;CHECK: sub {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
	%tmp3 = sub <1 x i64> %A, %B;
	ret <1 x i64> %tmp3
}

