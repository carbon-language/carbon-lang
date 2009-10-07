; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

%struct.__neon_int8x8x3_t = type { <8 x i8>,  <8 x i8>,  <8 x i8> }
%struct.__neon_int16x4x3_t = type { <4 x i16>, <4 x i16>, <4 x i16> }
%struct.__neon_int32x2x3_t = type { <2 x i32>, <2 x i32>, <2 x i32> }
%struct.__neon_float32x2x3_t = type { <2 x float>, <2 x float>, <2 x float> }

%struct.__neon_int8x16x3_t = type { <16 x i8>,  <16 x i8>,  <16 x i8> }
%struct.__neon_int16x8x3_t = type { <8 x i16>, <8 x i16>, <8 x i16> }
%struct.__neon_int32x4x3_t = type { <4 x i32>, <4 x i32>, <4 x i32> }
%struct.__neon_float32x4x3_t = type { <4 x float>, <4 x float>, <4 x float> }

define <8 x i8> @vld3i8(i8* %A) nounwind {
;CHECK: vld3i8:
;CHECK: vld3.8
	%tmp1 = call %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8(i8* %A)
        %tmp2 = extractvalue %struct.__neon_int8x8x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int8x8x3_t %tmp1, 2
        %tmp4 = add <8 x i8> %tmp2, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @vld3i16(i16* %A) nounwind {
;CHECK: vld3i16:
;CHECK: vld3.16
	%tmp1 = call %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3.v4i16(i16* %A)
        %tmp2 = extractvalue %struct.__neon_int16x4x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int16x4x3_t %tmp1, 2
        %tmp4 = add <4 x i16> %tmp2, %tmp3
	ret <4 x i16> %tmp4
}

define <2 x i32> @vld3i32(i32* %A) nounwind {
;CHECK: vld3i32:
;CHECK: vld3.32
	%tmp1 = call %struct.__neon_int32x2x3_t @llvm.arm.neon.vld3.v2i32(i32* %A)
        %tmp2 = extractvalue %struct.__neon_int32x2x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int32x2x3_t %tmp1, 2
        %tmp4 = add <2 x i32> %tmp2, %tmp3
	ret <2 x i32> %tmp4
}

define <2 x float> @vld3f(float* %A) nounwind {
;CHECK: vld3f:
;CHECK: vld3.32
	%tmp1 = call %struct.__neon_float32x2x3_t @llvm.arm.neon.vld3.v2f32(float* %A)
        %tmp2 = extractvalue %struct.__neon_float32x2x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_float32x2x3_t %tmp1, 2
        %tmp4 = add <2 x float> %tmp2, %tmp3
	ret <2 x float> %tmp4
}

define <16 x i8> @vld3Qi8(i8* %A) nounwind {
;CHECK: vld3Qi8:
;CHECK: vld3.8
;CHECK: vld3.8
	%tmp1 = call %struct.__neon_int8x16x3_t @llvm.arm.neon.vld3.v16i8(i8* %A)
        %tmp2 = extractvalue %struct.__neon_int8x16x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int8x16x3_t %tmp1, 2
        %tmp4 = add <16 x i8> %tmp2, %tmp3
	ret <16 x i8> %tmp4
}

define <8 x i16> @vld3Qi16(i16* %A) nounwind {
;CHECK: vld3Qi16:
;CHECK: vld3.16
;CHECK: vld3.16
	%tmp1 = call %struct.__neon_int16x8x3_t @llvm.arm.neon.vld3.v8i16(i16* %A)
        %tmp2 = extractvalue %struct.__neon_int16x8x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int16x8x3_t %tmp1, 2
        %tmp4 = add <8 x i16> %tmp2, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vld3Qi32(i32* %A) nounwind {
;CHECK: vld3Qi32:
;CHECK: vld3.32
;CHECK: vld3.32
	%tmp1 = call %struct.__neon_int32x4x3_t @llvm.arm.neon.vld3.v4i32(i32* %A)
        %tmp2 = extractvalue %struct.__neon_int32x4x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int32x4x3_t %tmp1, 2
        %tmp4 = add <4 x i32> %tmp2, %tmp3
	ret <4 x i32> %tmp4
}

define <4 x float> @vld3Qf(float* %A) nounwind {
;CHECK: vld3Qf:
;CHECK: vld3.32
;CHECK: vld3.32
	%tmp1 = call %struct.__neon_float32x4x3_t @llvm.arm.neon.vld3.v4f32(float* %A)
        %tmp2 = extractvalue %struct.__neon_float32x4x3_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_float32x4x3_t %tmp1, 2
        %tmp4 = add <4 x float> %tmp2, %tmp3
	ret <4 x float> %tmp4
}

declare %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8(i8*) nounwind readonly
declare %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3.v4i16(i8*) nounwind readonly
declare %struct.__neon_int32x2x3_t @llvm.arm.neon.vld3.v2i32(i8*) nounwind readonly
declare %struct.__neon_float32x2x3_t @llvm.arm.neon.vld3.v2f32(i8*) nounwind readonly

declare %struct.__neon_int8x16x3_t @llvm.arm.neon.vld3.v16i8(i8*) nounwind readonly
declare %struct.__neon_int16x8x3_t @llvm.arm.neon.vld3.v8i16(i8*) nounwind readonly
declare %struct.__neon_int32x4x3_t @llvm.arm.neon.vld3.v4i32(i8*) nounwind readonly
declare %struct.__neon_float32x4x3_t @llvm.arm.neon.vld3.v4f32(i8*) nounwind readonly
