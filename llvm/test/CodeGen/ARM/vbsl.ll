; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @v_bsli8(<8 x i8>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
;CHECK: v_bsli8:
;CHECK: vbsl
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
	%tmp4 = and <8 x i8> %tmp1, %tmp2
	%tmp5 = xor <8 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp6 = and <8 x i8> %tmp5, %tmp3
	%tmp7 = or <8 x i8> %tmp4, %tmp6
	ret <8 x i8> %tmp7
}

define <4 x i16> @v_bsli16(<4 x i16>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
;CHECK: v_bsli16:
;CHECK: vbsl
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
	%tmp4 = and <4 x i16> %tmp1, %tmp2
	%tmp5 = xor <4 x i16> %tmp1, < i16 -1, i16 -1, i16 -1, i16 -1 >
	%tmp6 = and <4 x i16> %tmp5, %tmp3
	%tmp7 = or <4 x i16> %tmp4, %tmp6
	ret <4 x i16> %tmp7
}

define <2 x i32> @v_bsli32(<2 x i32>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
;CHECK: v_bsli32:
;CHECK: vbsl
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
	%tmp4 = and <2 x i32> %tmp1, %tmp2
	%tmp5 = xor <2 x i32> %tmp1, < i32 -1, i32 -1 >
	%tmp6 = and <2 x i32> %tmp5, %tmp3
	%tmp7 = or <2 x i32> %tmp4, %tmp6
	ret <2 x i32> %tmp7
}

define <1 x i64> @v_bsli64(<1 x i64>* %A, <1 x i64>* %B, <1 x i64>* %C) nounwind {
;CHECK: v_bsli64:
;CHECK: vbsl
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = load <1 x i64>* %C
	%tmp4 = and <1 x i64> %tmp1, %tmp2
	%tmp5 = xor <1 x i64> %tmp1, < i64 -1 >
	%tmp6 = and <1 x i64> %tmp5, %tmp3
	%tmp7 = or <1 x i64> %tmp4, %tmp6
	ret <1 x i64> %tmp7
}

define <16 x i8> @v_bslQi8(<16 x i8>* %A, <16 x i8>* %B, <16 x i8>* %C) nounwind {
;CHECK: v_bslQi8:
;CHECK: vbsl
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = load <16 x i8>* %C
	%tmp4 = and <16 x i8> %tmp1, %tmp2
	%tmp5 = xor <16 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp6 = and <16 x i8> %tmp5, %tmp3
	%tmp7 = or <16 x i8> %tmp4, %tmp6
	ret <16 x i8> %tmp7
}

define <8 x i16> @v_bslQi16(<8 x i16>* %A, <8 x i16>* %B, <8 x i16>* %C) nounwind {
;CHECK: v_bslQi16:
;CHECK: vbsl
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = load <8 x i16>* %C
	%tmp4 = and <8 x i16> %tmp1, %tmp2
	%tmp5 = xor <8 x i16> %tmp1, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1 >
	%tmp6 = and <8 x i16> %tmp5, %tmp3
	%tmp7 = or <8 x i16> %tmp4, %tmp6
	ret <8 x i16> %tmp7
}

define <4 x i32> @v_bslQi32(<4 x i32>* %A, <4 x i32>* %B, <4 x i32>* %C) nounwind {
;CHECK: v_bslQi32:
;CHECK: vbsl
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = load <4 x i32>* %C
	%tmp4 = and <4 x i32> %tmp1, %tmp2
	%tmp5 = xor <4 x i32> %tmp1, < i32 -1, i32 -1, i32 -1, i32 -1 >
	%tmp6 = and <4 x i32> %tmp5, %tmp3
	%tmp7 = or <4 x i32> %tmp4, %tmp6
	ret <4 x i32> %tmp7
}

define <2 x i64> @v_bslQi64(<2 x i64>* %A, <2 x i64>* %B, <2 x i64>* %C) nounwind {
;CHECK: v_bslQi64:
;CHECK: vbsl
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = load <2 x i64>* %C
	%tmp4 = and <2 x i64> %tmp1, %tmp2
	%tmp5 = xor <2 x i64> %tmp1, < i64 -1, i64 -1 >
	%tmp6 = and <2 x i64> %tmp5, %tmp3
	%tmp7 = or <2 x i64> %tmp4, %tmp6
	ret <2 x i64> %tmp7
}
