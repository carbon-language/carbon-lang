; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vcges8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vcges8:
;CHECK: vcge.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = icmp sge <8 x i8> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <4 x i16> @vcges16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vcges16:
;CHECK: vcge.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = icmp sge <4 x i16> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <2 x i32> @vcges32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vcges32:
;CHECK: vcge.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = icmp sge <2 x i32> %tmp1, %tmp2
        %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <8 x i8> @vcgeu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vcgeu8:
;CHECK: vcge.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = icmp uge <8 x i8> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <4 x i16> @vcgeu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vcgeu16:
;CHECK: vcge.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = icmp uge <4 x i16> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <2 x i32> @vcgeu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vcgeu32:
;CHECK: vcge.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = icmp uge <2 x i32> %tmp1, %tmp2
        %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <2 x i32> @vcgef32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vcgef32:
;CHECK: vcge.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = fcmp oge <2 x float> %tmp1, %tmp2
        %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <16 x i8> @vcgeQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vcgeQs8:
;CHECK: vcge.s8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = icmp sge <16 x i8> %tmp1, %tmp2
        %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <8 x i16> @vcgeQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vcgeQs16:
;CHECK: vcge.s16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = icmp sge <8 x i16> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <4 x i32> @vcgeQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vcgeQs32:
;CHECK: vcge.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = icmp sge <4 x i32> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <16 x i8> @vcgeQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vcgeQu8:
;CHECK: vcge.u8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = icmp uge <16 x i8> %tmp1, %tmp2
        %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <8 x i16> @vcgeQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vcgeQu16:
;CHECK: vcge.u16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = icmp uge <8 x i16> %tmp1, %tmp2
        %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <4 x i32> @vcgeQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vcgeQu32:
;CHECK: vcge.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = icmp uge <4 x i32> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <4 x i32> @vcgeQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK: vcgeQf32:
;CHECK: vcge.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = fcmp oge <4 x float> %tmp1, %tmp2
        %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
