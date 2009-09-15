; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @v_eori8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: v_eori8:
;CHECK: veor
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = xor <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @v_eori16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: v_eori16:
;CHECK: veor
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = xor <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @v_eori32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: v_eori32:
;CHECK: veor
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = xor <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @v_eori64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: v_eori64:
;CHECK: veor
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = xor <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <16 x i8> @v_eorQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: v_eorQi8:
;CHECK: veor
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = xor <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @v_eorQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: v_eorQi16:
;CHECK: veor
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = xor <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @v_eorQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: v_eorQi32:
;CHECK: veor
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = xor <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @v_eorQi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: v_eorQi64:
;CHECK: veor
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = xor <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}
