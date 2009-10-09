; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @v_dup8(i8 %A) nounwind {
;CHECK: v_dup8:
;CHECK: vdup.8
	%tmp1 = insertelement <8 x i8> zeroinitializer, i8 %A, i32 0
	%tmp2 = insertelement <8 x i8> %tmp1, i8 %A, i32 1
	%tmp3 = insertelement <8 x i8> %tmp2, i8 %A, i32 2
	%tmp4 = insertelement <8 x i8> %tmp3, i8 %A, i32 3
	%tmp5 = insertelement <8 x i8> %tmp4, i8 %A, i32 4
	%tmp6 = insertelement <8 x i8> %tmp5, i8 %A, i32 5
	%tmp7 = insertelement <8 x i8> %tmp6, i8 %A, i32 6
	%tmp8 = insertelement <8 x i8> %tmp7, i8 %A, i32 7
	ret <8 x i8> %tmp8
}

define <4 x i16> @v_dup16(i16 %A) nounwind {
;CHECK: v_dup16:
;CHECK: vdup.16
	%tmp1 = insertelement <4 x i16> zeroinitializer, i16 %A, i32 0
	%tmp2 = insertelement <4 x i16> %tmp1, i16 %A, i32 1
	%tmp3 = insertelement <4 x i16> %tmp2, i16 %A, i32 2
	%tmp4 = insertelement <4 x i16> %tmp3, i16 %A, i32 3
	ret <4 x i16> %tmp4
}

define <2 x i32> @v_dup32(i32 %A) nounwind {
;CHECK: v_dup32:
;CHECK: vdup.32
	%tmp1 = insertelement <2 x i32> zeroinitializer, i32 %A, i32 0
	%tmp2 = insertelement <2 x i32> %tmp1, i32 %A, i32 1
	ret <2 x i32> %tmp2
}

define <2 x float> @v_dupfloat(float %A) nounwind {
;CHECK: v_dupfloat:
;CHECK: vdup.32
	%tmp1 = insertelement <2 x float> zeroinitializer, float %A, i32 0
	%tmp2 = insertelement <2 x float> %tmp1, float %A, i32 1
	ret <2 x float> %tmp2
}

define <16 x i8> @v_dupQ8(i8 %A) nounwind {
;CHECK: v_dupQ8:
;CHECK: vdup.8
	%tmp1 = insertelement <16 x i8> zeroinitializer, i8 %A, i32 0
	%tmp2 = insertelement <16 x i8> %tmp1, i8 %A, i32 1
	%tmp3 = insertelement <16 x i8> %tmp2, i8 %A, i32 2
	%tmp4 = insertelement <16 x i8> %tmp3, i8 %A, i32 3
	%tmp5 = insertelement <16 x i8> %tmp4, i8 %A, i32 4
	%tmp6 = insertelement <16 x i8> %tmp5, i8 %A, i32 5
	%tmp7 = insertelement <16 x i8> %tmp6, i8 %A, i32 6
	%tmp8 = insertelement <16 x i8> %tmp7, i8 %A, i32 7
	%tmp9 = insertelement <16 x i8> %tmp8, i8 %A, i32 8
	%tmp10 = insertelement <16 x i8> %tmp9, i8 %A, i32 9
	%tmp11 = insertelement <16 x i8> %tmp10, i8 %A, i32 10
	%tmp12 = insertelement <16 x i8> %tmp11, i8 %A, i32 11
	%tmp13 = insertelement <16 x i8> %tmp12, i8 %A, i32 12
	%tmp14 = insertelement <16 x i8> %tmp13, i8 %A, i32 13
	%tmp15 = insertelement <16 x i8> %tmp14, i8 %A, i32 14
	%tmp16 = insertelement <16 x i8> %tmp15, i8 %A, i32 15
	ret <16 x i8> %tmp16
}

define <8 x i16> @v_dupQ16(i16 %A) nounwind {
;CHECK: v_dupQ16:
;CHECK: vdup.16
	%tmp1 = insertelement <8 x i16> zeroinitializer, i16 %A, i32 0
	%tmp2 = insertelement <8 x i16> %tmp1, i16 %A, i32 1
	%tmp3 = insertelement <8 x i16> %tmp2, i16 %A, i32 2
	%tmp4 = insertelement <8 x i16> %tmp3, i16 %A, i32 3
	%tmp5 = insertelement <8 x i16> %tmp4, i16 %A, i32 4
	%tmp6 = insertelement <8 x i16> %tmp5, i16 %A, i32 5
	%tmp7 = insertelement <8 x i16> %tmp6, i16 %A, i32 6
	%tmp8 = insertelement <8 x i16> %tmp7, i16 %A, i32 7
	ret <8 x i16> %tmp8
}

define <4 x i32> @v_dupQ32(i32 %A) nounwind {
;CHECK: v_dupQ32:
;CHECK: vdup.32
	%tmp1 = insertelement <4 x i32> zeroinitializer, i32 %A, i32 0
	%tmp2 = insertelement <4 x i32> %tmp1, i32 %A, i32 1
	%tmp3 = insertelement <4 x i32> %tmp2, i32 %A, i32 2
	%tmp4 = insertelement <4 x i32> %tmp3, i32 %A, i32 3
	ret <4 x i32> %tmp4
}

define <4 x float> @v_dupQfloat(float %A) nounwind {
;CHECK: v_dupQfloat:
;CHECK: vdup.32
	%tmp1 = insertelement <4 x float> zeroinitializer, float %A, i32 0
	%tmp2 = insertelement <4 x float> %tmp1, float %A, i32 1
	%tmp3 = insertelement <4 x float> %tmp2, float %A, i32 2
	%tmp4 = insertelement <4 x float> %tmp3, float %A, i32 3
	ret <4 x float> %tmp4
}

; Check to make sure it works with shuffles, too.

define <8 x i8> @v_shuffledup8(i8 %A) nounwind {
;CHECK: v_shuffledup8:
;CHECK: vdup.8
	%tmp1 = insertelement <8 x i8> undef, i8 %A, i32 0
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> zeroinitializer
	ret <8 x i8> %tmp2
}

define <4 x i16> @v_shuffledup16(i16 %A) nounwind {
;CHECK: v_shuffledup16:
;CHECK: vdup.16
	%tmp1 = insertelement <4 x i16> undef, i16 %A, i32 0
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> zeroinitializer
	ret <4 x i16> %tmp2
}

define <2 x i32> @v_shuffledup32(i32 %A) nounwind {
;CHECK: v_shuffledup32:
;CHECK: vdup.32
	%tmp1 = insertelement <2 x i32> undef, i32 %A, i32 0
	%tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <2 x i32> zeroinitializer
	ret <2 x i32> %tmp2
}

define <2 x float> @v_shuffledupfloat(float %A) nounwind {
;CHECK: v_shuffledupfloat:
;CHECK: vdup.32
	%tmp1 = insertelement <2 x float> undef, float %A, i32 0
	%tmp2 = shufflevector <2 x float> %tmp1, <2 x float> undef, <2 x i32> zeroinitializer
	ret <2 x float> %tmp2
}

define <16 x i8> @v_shuffledupQ8(i8 %A) nounwind {
;CHECK: v_shuffledupQ8:
;CHECK: vdup.8
	%tmp1 = insertelement <16 x i8> undef, i8 %A, i32 0
	%tmp2 = shufflevector <16 x i8> %tmp1, <16 x i8> undef, <16 x i32> zeroinitializer
	ret <16 x i8> %tmp2
}

define <8 x i16> @v_shuffledupQ16(i16 %A) nounwind {
;CHECK: v_shuffledupQ16:
;CHECK: vdup.16
	%tmp1 = insertelement <8 x i16> undef, i16 %A, i32 0
	%tmp2 = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <8 x i32> zeroinitializer
	ret <8 x i16> %tmp2
}

define <4 x i32> @v_shuffledupQ32(i32 %A) nounwind {
;CHECK: v_shuffledupQ32:
;CHECK: vdup.32
	%tmp1 = insertelement <4 x i32> undef, i32 %A, i32 0
	%tmp2 = shufflevector <4 x i32> %tmp1, <4 x i32> undef, <4 x i32> zeroinitializer
	ret <4 x i32> %tmp2
}

define <4 x float> @v_shuffledupQfloat(float %A) nounwind {
;CHECK: v_shuffledupQfloat:
;CHECK: vdup.32
	%tmp1 = insertelement <4 x float> undef, float %A, i32 0
	%tmp2 = shufflevector <4 x float> %tmp1, <4 x float> undef, <4 x i32> zeroinitializer
	ret <4 x float> %tmp2
}

define <2 x float> @v_shuffledupfloat2(float* %A) nounwind {
;CHECK: v_shuffledupfloat2:
;CHECK: vdup.32
	%tmp0 = load float* %A
        %tmp1 = insertelement <2 x float> undef, float %tmp0, i32 0
        %tmp2 = shufflevector <2 x float> %tmp1, <2 x float> undef, <2 x i32> zeroinitializer
        ret <2 x float> %tmp2
}

define <4 x float> @v_shuffledupQfloat2(float* %A) nounwind {
;CHECK: v_shuffledupQfloat2:
;CHECK: vdup.32
        %tmp0 = load float* %A
        %tmp1 = insertelement <4 x float> undef, float %tmp0, i32 0
        %tmp2 = shufflevector <4 x float> %tmp1, <4 x float> undef, <4 x i32> zeroinitializer
        ret <4 x float> %tmp2
}

define <8 x i8> @vduplane8(<8 x i8>* %A) nounwind {
;CHECK: vduplane8:
;CHECK: vdup.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> < i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1 >
	ret <8 x i8> %tmp2
}

define <4 x i16> @vduplane16(<4 x i16>* %A) nounwind {
;CHECK: vduplane16:
;CHECK: vdup.16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> < i32 1, i32 1, i32 1, i32 1 >
	ret <4 x i16> %tmp2
}

define <2 x i32> @vduplane32(<2 x i32>* %A) nounwind {
;CHECK: vduplane32:
;CHECK: vdup.32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <2 x i32> < i32 1, i32 1 >
	ret <2 x i32> %tmp2
}

define <2 x float> @vduplanefloat(<2 x float>* %A) nounwind {
;CHECK: vduplanefloat:
;CHECK: vdup.32
	%tmp1 = load <2 x float>* %A
	%tmp2 = shufflevector <2 x float> %tmp1, <2 x float> undef, <2 x i32> < i32 1, i32 1 >
	ret <2 x float> %tmp2
}

define <16 x i8> @vduplaneQ8(<8 x i8>* %A) nounwind {
;CHECK: vduplaneQ8:
;CHECK: vdup.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <16 x i32> < i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1 >
	ret <16 x i8> %tmp2
}

define <8 x i16> @vduplaneQ16(<4 x i16>* %A) nounwind {
;CHECK: vduplaneQ16:
;CHECK: vdup.16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <8 x i32> < i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1 >
	ret <8 x i16> %tmp2
}

define <4 x i32> @vduplaneQ32(<2 x i32>* %A) nounwind {
;CHECK: vduplaneQ32:
;CHECK: vdup.32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <4 x i32> < i32 1, i32 1, i32 1, i32 1 >
	ret <4 x i32> %tmp2
}

define <4 x float> @vduplaneQfloat(<2 x float>* %A) nounwind {
;CHECK: vduplaneQfloat:
;CHECK: vdup.32
	%tmp1 = load <2 x float>* %A
	%tmp2 = shufflevector <2 x float> %tmp1, <2 x float> undef, <4 x i32> < i32 1, i32 1, i32 1, i32 1 >
	ret <4 x float> %tmp2
}

define arm_apcscc <2 x i64> @foo(<2 x i64> %arg0_int64x1_t) nounwind readnone {
entry:
  %0 = shufflevector <2 x i64> %arg0_int64x1_t, <2 x i64> undef, <2 x i32> <i32 1, i32 1>
  ret <2 x i64> %0
}

define arm_apcscc <2 x i64> @bar(<2 x i64> %arg0_int64x1_t) nounwind readnone {
entry:
  %0 = shufflevector <2 x i64> %arg0_int64x1_t, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  ret <2 x i64> %0
}

define arm_apcscc <2 x double> @baz(<2 x double> %arg0_int64x1_t) nounwind readnone {
entry:
  %0 = shufflevector <2 x double> %arg0_int64x1_t, <2 x double> undef, <2 x i32> <i32 1, i32 1>
  ret <2 x double> %0
}

define arm_apcscc <2 x double> @qux(<2 x double> %arg0_int64x1_t) nounwind readnone {
entry:
  %0 = shufflevector <2 x double> %arg0_int64x1_t, <2 x double> undef, <2 x i32> <i32 0, i32 0>
  ret <2 x double> %0
}
