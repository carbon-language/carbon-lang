; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vsubi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vsubi8:
;CHECK: vsub.i8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = sub <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vsubi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vsubi16:
;CHECK: vsub.i16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = sub <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vsubi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vsubi32:
;CHECK: vsub.i32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = sub <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @vsubi64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: vsubi64:
;CHECK: vsub.i64
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = sub <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <2 x float> @vsubf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vsubf32:
;CHECK: vsub.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = sub <2 x float> %tmp1, %tmp2
	ret <2 x float> %tmp3
}

define <16 x i8> @vsubQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vsubQi8:
;CHECK: vsub.i8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = sub <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vsubQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vsubQi16:
;CHECK: vsub.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = sub <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vsubQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vsubQi32:
;CHECK: vsub.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = sub <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @vsubQi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vsubQi64:
;CHECK: vsub.i64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = sub <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

define <4 x float> @vsubQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK: vsubQf32:
;CHECK: vsub.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = sub <4 x float> %tmp1, %tmp2
	ret <4 x float> %tmp3
}
