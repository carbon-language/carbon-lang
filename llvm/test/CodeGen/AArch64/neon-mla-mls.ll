; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-none-linux-gnu -mattr=+neon | FileCheck %s


define <8 x i8> @mla8xi8(<8 x i8> %A, <8 x i8> %B, <8 x i8> %C) {
;CHECK: mla {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp1 = mul <8 x i8> %A, %B;
	%tmp2 = add <8 x i8> %C, %tmp1;
	ret <8 x i8> %tmp2
}

define <16 x i8> @mla16xi8(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C) {
;CHECK: mla {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp1 = mul <16 x i8> %A, %B;
	%tmp2 = add <16 x i8> %C, %tmp1;
	ret <16 x i8> %tmp2
}

define <4 x i16> @mla4xi16(<4 x i16> %A, <4 x i16> %B, <4 x i16> %C) {
;CHECK: mla {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp1 = mul <4 x i16> %A, %B;
	%tmp2 = add <4 x i16> %C, %tmp1;
	ret <4 x i16> %tmp2
}

define <8 x i16> @mla8xi16(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C) {
;CHECK: mla {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp1 = mul <8 x i16> %A, %B;
	%tmp2 = add <8 x i16> %C, %tmp1;
	ret <8 x i16> %tmp2
}

define <2 x i32> @mla2xi32(<2 x i32> %A, <2 x i32> %B, <2 x i32> %C) {
;CHECK: mla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp1 = mul <2 x i32> %A, %B;
	%tmp2 = add <2 x i32> %C, %tmp1;
	ret <2 x i32> %tmp2
}

define <4 x i32> @mla4xi32(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C) {
;CHECK: mla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp1 = mul <4 x i32> %A, %B;
	%tmp2 = add <4 x i32> %C, %tmp1;
	ret <4 x i32> %tmp2
}

define <8 x i8> @mls8xi8(<8 x i8> %A, <8 x i8> %B, <8 x i8> %C) {
;CHECK: mls {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp1 = mul <8 x i8> %A, %B;
	%tmp2 = sub <8 x i8> %C, %tmp1;
	ret <8 x i8> %tmp2
}

define <16 x i8> @mls16xi8(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C) {
;CHECK: mls {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp1 = mul <16 x i8> %A, %B;
	%tmp2 = sub <16 x i8> %C, %tmp1;
	ret <16 x i8> %tmp2
}

define <4 x i16> @mls4xi16(<4 x i16> %A, <4 x i16> %B, <4 x i16> %C) {
;CHECK: mls {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp1 = mul <4 x i16> %A, %B;
	%tmp2 = sub <4 x i16> %C, %tmp1;
	ret <4 x i16> %tmp2
}

define <8 x i16> @mls8xi16(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C) {
;CHECK: mls {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp1 = mul <8 x i16> %A, %B;
	%tmp2 = sub <8 x i16> %C, %tmp1;
	ret <8 x i16> %tmp2
}

define <2 x i32> @mls2xi32(<2 x i32> %A, <2 x i32> %B, <2 x i32> %C) {
;CHECK: mls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp1 = mul <2 x i32> %A, %B;
	%tmp2 = sub <2 x i32> %C, %tmp1;
	ret <2 x i32> %tmp2
}

define <4 x i32> @mls4xi32(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C) {
;CHECK: mls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp1 = mul <4 x i32> %A, %B;
	%tmp2 = sub <4 x i32> %C, %tmp1;
	ret <4 x i32> %tmp2
}


