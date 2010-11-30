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

%struct.__neon_int8x8x2_t = type { <8 x i8>, <8 x i8> }
%struct.__neon_int2x32x2_t = type { <2 x i32>, <2 x i32> }

define <8 x i8> @vld2dupi8(i8* %A) nounwind {
;CHECK: vld2dupi8:
;Check the (default) alignment value.
;CHECK: vld2.8 {d16[], d17[]}, [r0]
	%tmp0 = tail call %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2lane.v8i8(i8* %A, <8 x i8> undef, <8 x i8> undef, i32 0, i32 1)
	%tmp1 = extractvalue %struct.__neon_int8x8x2_t %tmp0, 0
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> zeroinitializer
	%tmp3 = extractvalue %struct.__neon_int8x8x2_t %tmp0, 1
	%tmp4 = shufflevector <8 x i8> %tmp3, <8 x i8> undef, <8 x i32> zeroinitializer
        %tmp5 = add <8 x i8> %tmp2, %tmp4
        ret <8 x i8> %tmp5
}

define <2 x i32> @vld2dupi32(i32* %A) nounwind {
;CHECK: vld2dupi32:
;Check the alignment value.  Max for this instruction is 64 bits:
;CHECK: vld2.32 {d16[], d17[]}, [r0, :64]
	%tmp0 = tail call %struct.__neon_int2x32x2_t @llvm.arm.neon.vld2lane.v2i32(i32* %A, <2 x i32> undef, <2 x i32> undef, i32 0, i32 16)
	%tmp1 = extractvalue %struct.__neon_int2x32x2_t %tmp0, 0
	%tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <2 x i32> zeroinitializer
	%tmp3 = extractvalue %struct.__neon_int2x32x2_t %tmp0, 1
	%tmp4 = shufflevector <2 x i32> %tmp3, <2 x i32> undef, <2 x i32> zeroinitializer
        %tmp5 = add <2 x i32> %tmp2, %tmp4
        ret <2 x i32> %tmp5
}

declare %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2lane.v8i8(i8*, <8 x i8>, <8 x i8>, i32, i32) nounwind readonly
declare %struct.__neon_int2x32x2_t @llvm.arm.neon.vld2lane.v2i32(i32*, <2 x i32>, <2 x i32>, i32, i32) nounwind readonly

%struct.__neon_int16x4x3_t = type { <4 x i16>, <4 x i16>, <4 x i16> }

define <4 x i16> @vld3dupi16(i16* %A) nounwind {
;CHECK: vld3dupi16:
;Check the (default) alignment value. VLD3 does not support alignment.
;CHECK: vld3.16 {d16[], d17[], d18[]}, [r0]
	%tmp0 = tail call %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3lane.v4i16(i16* %A, <4 x i16> undef, <4 x i16> undef, <4 x i16> undef, i32 0, i32 8)
	%tmp1 = extractvalue %struct.__neon_int16x4x3_t %tmp0, 0
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> zeroinitializer
	%tmp3 = extractvalue %struct.__neon_int16x4x3_t %tmp0, 1
	%tmp4 = shufflevector <4 x i16> %tmp3, <4 x i16> undef, <4 x i32> zeroinitializer
	%tmp5 = extractvalue %struct.__neon_int16x4x3_t %tmp0, 2
	%tmp6 = shufflevector <4 x i16> %tmp5, <4 x i16> undef, <4 x i32> zeroinitializer
        %tmp7 = add <4 x i16> %tmp2, %tmp4
        %tmp8 = add <4 x i16> %tmp7, %tmp6
        ret <4 x i16> %tmp8
}

declare %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3lane.v4i16(i16*, <4 x i16>, <4 x i16>, <4 x i16>, i32, i32) nounwind readonly

%struct.__neon_int32x2x4_t = type { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }

define <2 x i32> @vld4dupi32(i32* %A) nounwind {
;CHECK: vld4dupi32:
;Check the alignment value.  Max for this instruction is 128 bits:
;CHECK: vld4.32 {d16[], d17[], d18[], d19[]}, [r0, :128]
	%tmp0 = tail call %struct.__neon_int32x2x4_t @llvm.arm.neon.vld4lane.v2i32(i32* %A, <2 x i32> undef, <2 x i32> undef, <2 x i32> undef, <2 x i32> undef, i32 0, i32 32)
	%tmp1 = extractvalue %struct.__neon_int32x2x4_t %tmp0, 0
	%tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <2 x i32> zeroinitializer
	%tmp3 = extractvalue %struct.__neon_int32x2x4_t %tmp0, 1
	%tmp4 = shufflevector <2 x i32> %tmp3, <2 x i32> undef, <2 x i32> zeroinitializer
	%tmp5 = extractvalue %struct.__neon_int32x2x4_t %tmp0, 2
	%tmp6 = shufflevector <2 x i32> %tmp5, <2 x i32> undef, <2 x i32> zeroinitializer
	%tmp7 = extractvalue %struct.__neon_int32x2x4_t %tmp0, 3
	%tmp8 = shufflevector <2 x i32> %tmp7, <2 x i32> undef, <2 x i32> zeroinitializer
        %tmp9 = add <2 x i32> %tmp2, %tmp4
        %tmp10 = add <2 x i32> %tmp6, %tmp8
        %tmp11 = add <2 x i32> %tmp9, %tmp10
        ret <2 x i32> %tmp11
}

declare %struct.__neon_int32x2x4_t @llvm.arm.neon.vld4lane.v2i32(i32*, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32) nounwind readonly
