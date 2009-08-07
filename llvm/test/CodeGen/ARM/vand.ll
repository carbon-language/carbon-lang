; RUN: llvm-as < %s | llc -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @v_andi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: v_andi8:
;CHECK: vand
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = and <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @v_andi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: v_andi16:
;CHECK: vand
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = and <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @v_andi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: v_andi32:
;CHECK: vand
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = and <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @v_andi64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK: v_andi64:
;CHECK: vand
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = and <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <16 x i8> @v_andQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: v_andQi8:
;CHECK: vand
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = and <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @v_andQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: v_andQi16:
;CHECK: vand
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = and <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @v_andQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: v_andQi32:
;CHECK: vand
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = and <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @v_andQi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: v_andQi64:
;CHECK: vand
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = and <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}
