; RUN: llc < %s -march=arm -mattr=+neon > %t
; RUN: grep {vmov\\.8} %t | count 2
; RUN: grep {vmov\\.16} %t | count 2
; RUN: grep {vmov\\.32} %t | count 2
; RUN: grep {fcpys} %t | count 2

define <8 x i8> @vset_lane8(<8 x i8>* %A, i8 %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = insertelement <8 x i8> %tmp1, i8 %B, i32 1
	ret <8 x i8> %tmp2
}

define <4 x i16> @vset_lane16(<4 x i16>* %A, i16 %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = insertelement <4 x i16> %tmp1, i16 %B, i32 1
	ret <4 x i16> %tmp2
}

define <2 x i32> @vset_lane32(<2 x i32>* %A, i32 %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = insertelement <2 x i32> %tmp1, i32 %B, i32 1
	ret <2 x i32> %tmp2
}

define <16 x i8> @vsetQ_lane8(<16 x i8>* %A, i8 %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = insertelement <16 x i8> %tmp1, i8 %B, i32 1
	ret <16 x i8> %tmp2
}

define <8 x i16> @vsetQ_lane16(<8 x i16>* %A, i16 %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = insertelement <8 x i16> %tmp1, i16 %B, i32 1
	ret <8 x i16> %tmp2
}

define <4 x i32> @vsetQ_lane32(<4 x i32>* %A, i32 %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = insertelement <4 x i32> %tmp1, i32 %B, i32 1
	ret <4 x i32> %tmp2
}

define arm_aapcs_vfpcc <2 x float> @test_vset_lanef32(float %arg0_float32_t, <2 x float> %arg1_float32x2_t) nounwind {
entry:
  %0 = insertelement <2 x float> %arg1_float32x2_t, float %arg0_float32_t, i32 1 ; <<2 x float>> [#uses=1]
  ret <2 x float> %0
}
