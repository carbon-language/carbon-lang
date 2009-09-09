; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vcgts8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vcgts8:
;CHECK: vcgt.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = icmp sgt <8 x i8> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <4 x i16> @vcgts16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vcgts16:
;CHECK: vcgt.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = icmp sgt <4 x i16> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <2 x i32> @vcgts32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vcgts32:
;CHECK: vcgt.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = icmp sgt <2 x i32> %tmp1, %tmp2
        %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <8 x i8> @vcgtu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vcgtu8:
;CHECK: vcgt.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = icmp ugt <8 x i8> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <4 x i16> @vcgtu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vcgtu16:
;CHECK: vcgt.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = icmp ugt <4 x i16> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <2 x i32> @vcgtu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vcgtu32:
;CHECK: vcgt.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = icmp ugt <2 x i32> %tmp1, %tmp2
        %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <2 x i32> @vcgtf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vcgtf32:
;CHECK: vcgt.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = fcmp ogt <2 x float> %tmp1, %tmp2
        %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <16 x i8> @vcgtQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vcgtQs8:
;CHECK: vcgt.s8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = icmp sgt <16 x i8> %tmp1, %tmp2
        %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <8 x i16> @vcgtQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vcgtQs16:
;CHECK: vcgt.s16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = icmp sgt <8 x i16> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <4 x i32> @vcgtQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vcgtQs32:
;CHECK: vcgt.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = icmp sgt <4 x i32> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <16 x i8> @vcgtQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vcgtQu8:
;CHECK: vcgt.u8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = icmp ugt <16 x i8> %tmp1, %tmp2
        %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <8 x i16> @vcgtQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vcgtQu16:
;CHECK: vcgt.u16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = icmp ugt <8 x i16> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <4 x i32> @vcgtQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vcgtQu32:
;CHECK: vcgt.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = icmp ugt <4 x i32> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <4 x i32> @vcgtQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK: vcgtQf32:
;CHECK: vcgt.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = fcmp ogt <4 x float> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
