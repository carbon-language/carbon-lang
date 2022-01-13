; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o -| FileCheck %s
; RUN: llc -mtriple=arm-eabi -mattr=+neon -regalloc=basic %s -o - | FileCheck %s

%struct.__neon_int8x8x3_t = type { <8 x i8>,  <8 x i8>,  <8 x i8> }
%struct.__neon_int16x4x3_t = type { <4 x i16>, <4 x i16>, <4 x i16> }
%struct.__neon_int32x2x3_t = type { <2 x i32>, <2 x i32>, <2 x i32> }
%struct.__neon_float32x2x3_t = type { <2 x float>, <2 x float>, <2 x float> }
%struct.__neon_int64x1x3_t = type { <1 x i64>, <1 x i64>, <1 x i64> }

%struct.__neon_int8x16x3_t = type { <16 x i8>,  <16 x i8>,  <16 x i8> }
%struct.__neon_int16x8x3_t = type { <8 x i16>, <8 x i16>, <8 x i16> }
%struct.__neon_int32x4x3_t = type { <4 x i32>, <4 x i32>, <4 x i32> }
%struct.__neon_float32x4x3_t = type { <4 x float>, <4 x float>, <4 x float> }

define <8 x i8> @vld3i8(i8* %A) nounwind {
;CHECK-LABEL: vld3i8:
;Check the alignment value.  Max for this instruction is 64 bits:
;CHECK: vld3.8 {d16, d17, d18}, [{{r[0-9]+|lr}}:64]
	%tmp1 = call %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8.p0i8(i8* %A, i32 32)
        %tmp2 = extractvalue %struct.__neon_int8x8x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int8x8x3_t %tmp1, 2
        %tmp4 = add <8 x i8> %tmp2, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @vld3i16(i16* %A) nounwind {
;CHECK-LABEL: vld3i16:
;CHECK: vld3.16
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = call %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3.v4i16.p0i8(i8* %tmp0, i32 1)
        %tmp2 = extractvalue %struct.__neon_int16x4x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int16x4x3_t %tmp1, 2
        %tmp4 = add <4 x i16> %tmp2, %tmp3
	ret <4 x i16> %tmp4
}

;Check for a post-increment updating load with register increment.
define <4 x i16> @vld3i16_update(i16** %ptr, i32 %inc) nounwind {
;CHECK-LABEL: vld3i16_update:
;CHECK: vld3.16 {d16, d17, d18}, [{{r[0-9]+|lr}}], {{r[0-9]+|lr}}
	%A = load i16*, i16** %ptr
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = call %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3.v4i16.p0i8(i8* %tmp0, i32 1)
	%tmp2 = extractvalue %struct.__neon_int16x4x3_t %tmp1, 0
	%tmp3 = extractvalue %struct.__neon_int16x4x3_t %tmp1, 2
	%tmp4 = add <4 x i16> %tmp2, %tmp3
	%tmp5 = getelementptr i16, i16* %A, i32 %inc
	store i16* %tmp5, i16** %ptr
	ret <4 x i16> %tmp4
}

define <2 x i32> @vld3i32(i32* %A) nounwind {
;CHECK-LABEL: vld3i32:
;CHECK: vld3.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = call %struct.__neon_int32x2x3_t @llvm.arm.neon.vld3.v2i32.p0i8(i8* %tmp0, i32 1)
        %tmp2 = extractvalue %struct.__neon_int32x2x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int32x2x3_t %tmp1, 2
        %tmp4 = add <2 x i32> %tmp2, %tmp3
	ret <2 x i32> %tmp4
}

define <2 x float> @vld3f(float* %A) nounwind {
;CHECK-LABEL: vld3f:
;CHECK: vld3.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = call %struct.__neon_float32x2x3_t @llvm.arm.neon.vld3.v2f32.p0i8(i8* %tmp0, i32 1)
        %tmp2 = extractvalue %struct.__neon_float32x2x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_float32x2x3_t %tmp1, 2
        %tmp4 = fadd <2 x float> %tmp2, %tmp3
	ret <2 x float> %tmp4
}

define <1 x i64> @vld3i64(i64* %A) nounwind {
;CHECK-LABEL: vld3i64:
;Check the alignment value.  Max for this instruction is 64 bits:
;CHECK: vld1.64 {d16, d17, d18}, [{{r[0-9]+|lr}}:64]
	%tmp0 = bitcast i64* %A to i8*
	%tmp1 = call %struct.__neon_int64x1x3_t @llvm.arm.neon.vld3.v1i64.p0i8(i8* %tmp0, i32 16)
        %tmp2 = extractvalue %struct.__neon_int64x1x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int64x1x3_t %tmp1, 2
        %tmp4 = add <1 x i64> %tmp2, %tmp3
	ret <1 x i64> %tmp4
}

define <1 x i64> @vld3i64_update(i64** %ptr, i64* %A) nounwind {
;CHECK-LABEL: vld3i64_update:
;CHECK: vld1.64	{d16, d17, d18}, [{{r[0-9]+|lr}}:64]!
        %tmp0 = bitcast i64* %A to i8*
        %tmp1 = call %struct.__neon_int64x1x3_t @llvm.arm.neon.vld3.v1i64.p0i8(i8* %tmp0, i32 16)
        %tmp5 = getelementptr i64, i64* %A, i32 3
        store i64* %tmp5, i64** %ptr
        %tmp2 = extractvalue %struct.__neon_int64x1x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int64x1x3_t %tmp1, 2
        %tmp4 = add <1 x i64> %tmp2, %tmp3
        ret <1 x i64> %tmp4
}

define <1 x i64> @vld3i64_reg_update(i64** %ptr, i64* %A) nounwind {
;CHECK-LABEL: vld3i64_reg_update:
;CHECK: vld1.64	{d16, d17, d18}, [{{r[0-9]+|lr}}:64], {{r[0-9]+|lr}}
        %tmp0 = bitcast i64* %A to i8*
        %tmp1 = call %struct.__neon_int64x1x3_t @llvm.arm.neon.vld3.v1i64.p0i8(i8* %tmp0, i32 16)
        %tmp5 = getelementptr i64, i64* %A, i32 1
        store i64* %tmp5, i64** %ptr
        %tmp2 = extractvalue %struct.__neon_int64x1x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int64x1x3_t %tmp1, 2
        %tmp4 = add <1 x i64> %tmp2, %tmp3
        ret <1 x i64> %tmp4
}

define <16 x i8> @vld3Qi8(i8* %A) nounwind {
;CHECK-LABEL: vld3Qi8:
;Check the alignment value.  Max for this instruction is 64 bits:
;CHECK: vld3.8 {d16, d18, d20}, [{{r[0-9]+|lr}}:64]!
;CHECK: vld3.8 {d17, d19, d21}, [{{r[0-9]+|lr}}:64]
	%tmp1 = call %struct.__neon_int8x16x3_t @llvm.arm.neon.vld3.v16i8.p0i8(i8* %A, i32 32)
        %tmp2 = extractvalue %struct.__neon_int8x16x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int8x16x3_t %tmp1, 2
        %tmp4 = add <16 x i8> %tmp2, %tmp3
	ret <16 x i8> %tmp4
}

define <8 x i16> @vld3Qi16(i16* %A) nounwind {
;CHECK-LABEL: vld3Qi16:
;CHECK: vld3.16
;CHECK: vld3.16
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = call %struct.__neon_int16x8x3_t @llvm.arm.neon.vld3.v8i16.p0i8(i8* %tmp0, i32 1)
        %tmp2 = extractvalue %struct.__neon_int16x8x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int16x8x3_t %tmp1, 2
        %tmp4 = add <8 x i16> %tmp2, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vld3Qi32(i32* %A) nounwind {
;CHECK-LABEL: vld3Qi32:
;CHECK: vld3.32
;CHECK: vld3.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = call %struct.__neon_int32x4x3_t @llvm.arm.neon.vld3.v4i32.p0i8(i8* %tmp0, i32 1)
        %tmp2 = extractvalue %struct.__neon_int32x4x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int32x4x3_t %tmp1, 2
        %tmp4 = add <4 x i32> %tmp2, %tmp3
	ret <4 x i32> %tmp4
}

;Check for a post-increment updating load. 
define <4 x i32> @vld3Qi32_update(i32** %ptr) nounwind {
;CHECK-LABEL: vld3Qi32_update:
;CHECK: vld3.32 {d16, d18, d20}, {{\[}}[[R:r[0-9]+|lr]]]!
;CHECK: vld3.32 {d17, d19, d21}, {{\[}}[[R]]]!
	%A = load i32*, i32** %ptr
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = call %struct.__neon_int32x4x3_t @llvm.arm.neon.vld3.v4i32.p0i8(i8* %tmp0, i32 1)
	%tmp2 = extractvalue %struct.__neon_int32x4x3_t %tmp1, 0
	%tmp3 = extractvalue %struct.__neon_int32x4x3_t %tmp1, 2
	%tmp4 = add <4 x i32> %tmp2, %tmp3
	%tmp5 = getelementptr i32, i32* %A, i32 12
	store i32* %tmp5, i32** %ptr
	ret <4 x i32> %tmp4
}

define <4 x float> @vld3Qf(float* %A) nounwind {
;CHECK-LABEL: vld3Qf:
;CHECK: vld3.32
;CHECK: vld3.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = call %struct.__neon_float32x4x3_t @llvm.arm.neon.vld3.v4f32.p0i8(i8* %tmp0, i32 1)
        %tmp2 = extractvalue %struct.__neon_float32x4x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_float32x4x3_t %tmp1, 2
        %tmp4 = fadd <4 x float> %tmp2, %tmp3
	ret <4 x float> %tmp4
}

declare %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8.p0i8(i8*, i32) nounwind readonly
declare %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3.v4i16.p0i8(i8*, i32) nounwind readonly
declare %struct.__neon_int32x2x3_t @llvm.arm.neon.vld3.v2i32.p0i8(i8*, i32) nounwind readonly
declare %struct.__neon_float32x2x3_t @llvm.arm.neon.vld3.v2f32.p0i8(i8*, i32) nounwind readonly
declare %struct.__neon_int64x1x3_t @llvm.arm.neon.vld3.v1i64.p0i8(i8*, i32) nounwind readonly

declare %struct.__neon_int8x16x3_t @llvm.arm.neon.vld3.v16i8.p0i8(i8*, i32) nounwind readonly
declare %struct.__neon_int16x8x3_t @llvm.arm.neon.vld3.v8i16.p0i8(i8*, i32) nounwind readonly
declare %struct.__neon_int32x4x3_t @llvm.arm.neon.vld3.v4i32.p0i8(i8*, i32) nounwind readonly
declare %struct.__neon_float32x4x3_t @llvm.arm.neon.vld3.v4f32.p0i8(i8*, i32) nounwind readonly
