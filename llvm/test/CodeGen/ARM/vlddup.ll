; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vld1dupi8(i8* %A) nounwind {
;CHECK: vld1dupi8:
;Check the (default) alignment value.
;CHECK: vld1.8 {d16[]}, [r0]
	%tmp1 = load i8* %A, align 8
	%tmp2 = insertelement <8 x i8> undef, i8 %tmp1, i32 0
	%tmp3 = shufflevector <8 x i8> %tmp2, <8 x i8> undef, <8 x i32> zeroinitializer
        ret <8 x i8> %tmp3
}

define <4 x i16> @vld1dupi16(i16* %A) nounwind {
;CHECK: vld1dupi16:
;Check the alignment value.  Max for this instruction is 16 bits:
;CHECK: vld1.16 {d16[]}, [r0, :16]
	%tmp1 = load i16* %A, align 8
	%tmp2 = insertelement <4 x i16> undef, i16 %tmp1, i32 0
	%tmp3 = shufflevector <4 x i16> %tmp2, <4 x i16> undef, <4 x i32> zeroinitializer
        ret <4 x i16> %tmp3
}

define <2 x i32> @vld1dupi32(i32* %A) nounwind {
;CHECK: vld1dupi32:
;Check the alignment value.  Max for this instruction is 32 bits:
;CHECK: vld1.32 {d16[]}, [r0, :32]
	%tmp1 = load i32* %A, align 8
	%tmp2 = insertelement <2 x i32> undef, i32 %tmp1, i32 0
	%tmp3 = shufflevector <2 x i32> %tmp2, <2 x i32> undef, <2 x i32> zeroinitializer
        ret <2 x i32> %tmp3
}

define <16 x i8> @vld1dupQi8(i8* %A) nounwind {
;CHECK: vld1dupQi8:
;Check the (default) alignment value.
;CHECK: vld1.8 {d16[], d17[]}, [r0]
	%tmp1 = load i8* %A, align 8
	%tmp2 = insertelement <16 x i8> undef, i8 %tmp1, i32 0
	%tmp3 = shufflevector <16 x i8> %tmp2, <16 x i8> undef, <16 x i32> zeroinitializer
        ret <16 x i8> %tmp3
}
