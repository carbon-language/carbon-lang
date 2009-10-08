; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vnegs8(<8 x i8>* %A) nounwind {
;CHECK: vnegs8:
;CHECK: vneg.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = sub <8 x i8> zeroinitializer, %tmp1
	ret <8 x i8> %tmp2
}

define <4 x i16> @vnegs16(<4 x i16>* %A) nounwind {
;CHECK: vnegs16:
;CHECK: vneg.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = sub <4 x i16> zeroinitializer, %tmp1
	ret <4 x i16> %tmp2
}

define <2 x i32> @vnegs32(<2 x i32>* %A) nounwind {
;CHECK: vnegs32:
;CHECK: vneg.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = sub <2 x i32> zeroinitializer, %tmp1
	ret <2 x i32> %tmp2
}

define <2 x float> @vnegf32(<2 x float>* %A) nounwind {
;CHECK: vnegf32:
;CHECK: vneg.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = sub <2 x float> < float -0.000000e+00, float -0.000000e+00 >, %tmp1
	ret <2 x float> %tmp2
}

define <16 x i8> @vnegQs8(<16 x i8>* %A) nounwind {
;CHECK: vnegQs8:
;CHECK: vneg.s8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = sub <16 x i8> zeroinitializer, %tmp1
	ret <16 x i8> %tmp2
}

define <8 x i16> @vnegQs16(<8 x i16>* %A) nounwind {
;CHECK: vnegQs16:
;CHECK: vneg.s16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = sub <8 x i16> zeroinitializer, %tmp1
	ret <8 x i16> %tmp2
}

define <4 x i32> @vnegQs32(<4 x i32>* %A) nounwind {
;CHECK: vnegQs32:
;CHECK: vneg.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = sub <4 x i32> zeroinitializer, %tmp1
	ret <4 x i32> %tmp2
}

define <4 x float> @vnegQf32(<4 x float>* %A) nounwind {
;CHECK: vnegQf32:
;CHECK: vneg.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = sub <4 x float> < float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00 >, %tmp1
	ret <4 x float> %tmp2
}
