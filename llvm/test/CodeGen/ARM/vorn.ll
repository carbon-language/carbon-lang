; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @v_orni8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: v_orni8:
;CHECK: vorn
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = xor <8 x i8> %tmp2, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp4 = or <8 x i8> %tmp1, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @v_orni16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: v_orni16:
;CHECK: vorn
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = xor <4 x i16> %tmp2, < i16 -1, i16 -1, i16 -1, i16 -1 >
	%tmp4 = or <4 x i16> %tmp1, %tmp3
	ret <4 x i16> %tmp4
}

define <2 x i32> @v_orni32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: v_orni32:
;CHECK: vorn
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = xor <2 x i32> %tmp2, < i32 -1, i32 -1 >
	%tmp4 = or <2 x i32> %tmp1, %tmp3
	ret <2 x i32> %tmp4
}

define <1 x i64> @v_orni64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: v_orni64:
;CHECK: vorn
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = xor <1 x i64> %tmp2, < i64 -1 >
	%tmp4 = or <1 x i64> %tmp1, %tmp3
	ret <1 x i64> %tmp4
}

define <16 x i8> @v_ornQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: v_ornQi8:
;CHECK: vorn
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = xor <16 x i8> %tmp2, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp4 = or <16 x i8> %tmp1, %tmp3
	ret <16 x i8> %tmp4
}

define <8 x i16> @v_ornQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: v_ornQi16:
;CHECK: vorn
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = xor <8 x i16> %tmp2, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1 >
	%tmp4 = or <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @v_ornQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: v_ornQi32:
;CHECK: vorn
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = xor <4 x i32> %tmp2, < i32 -1, i32 -1, i32 -1, i32 -1 >
	%tmp4 = or <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @v_ornQi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: v_ornQi64:
;CHECK: vorn
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = xor <2 x i64> %tmp2, < i64 -1, i64 -1 >
	%tmp4 = or <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}
