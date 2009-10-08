; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

%struct.__neon_int8x8x2_t = type { <8 x i8>,  <8 x i8> }
%struct.__neon_int16x4x2_t = type { <4 x i16>, <4 x i16> }
%struct.__neon_int32x2x2_t = type { <2 x i32>, <2 x i32> }
%struct.__neon_float32x2x2_t = type { <2 x float>, <2 x float> }

%struct.__neon_int16x8x2_t = type { <8 x i16>, <8 x i16> }
%struct.__neon_int32x4x2_t = type { <4 x i32>, <4 x i32> }
%struct.__neon_float32x4x2_t = type { <4 x float>, <4 x float> }

define <8 x i8> @vld2lanei8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK: vld2lanei8:
;CHECK: vld2.8
	%tmp1 = load <8 x i8>* %B
	%tmp2 = call %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2lane.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 1
        %tmp5 = add <8 x i8> %tmp3, %tmp4
	ret <8 x i8> %tmp5
}

define <4 x i16> @vld2lanei16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK: vld2lanei16:
;CHECK: vld2.16
	%tmp1 = load <4 x i16>* %B
	%tmp2 = call %struct.__neon_int16x4x2_t @llvm.arm.neon.vld2lane.v4i16(i16* %A, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__neon_int16x4x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int16x4x2_t %tmp2, 1
        %tmp5 = add <4 x i16> %tmp3, %tmp4
	ret <4 x i16> %tmp5
}

define <2 x i32> @vld2lanei32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK: vld2lanei32:
;CHECK: vld2.32
	%tmp1 = load <2 x i32>* %B
	%tmp2 = call %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2lane.v2i32(i32* %A, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__neon_int32x2x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int32x2x2_t %tmp2, 1
        %tmp5 = add <2 x i32> %tmp3, %tmp4
	ret <2 x i32> %tmp5
}

define <2 x float> @vld2lanef(float* %A, <2 x float>* %B) nounwind {
;CHECK: vld2lanef:
;CHECK: vld2.32
	%tmp1 = load <2 x float>* %B
	%tmp2 = call %struct.__neon_float32x2x2_t @llvm.arm.neon.vld2lane.v2f32(float* %A, <2 x float> %tmp1, <2 x float> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__neon_float32x2x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_float32x2x2_t %tmp2, 1
        %tmp5 = add <2 x float> %tmp3, %tmp4
	ret <2 x float> %tmp5
}

define <8 x i16> @vld2laneQi16(i16* %A, <8 x i16>* %B) nounwind {
;CHECK: vld2laneQi16:
;CHECK: vld2.16
	%tmp1 = load <8 x i16>* %B
	%tmp2 = call %struct.__neon_int16x8x2_t @llvm.arm.neon.vld2lane.v8i16(i16* %A, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__neon_int16x8x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int16x8x2_t %tmp2, 1
        %tmp5 = add <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

define <4 x i32> @vld2laneQi32(i32* %A, <4 x i32>* %B) nounwind {
;CHECK: vld2laneQi32:
;CHECK: vld2.32
	%tmp1 = load <4 x i32>* %B
	%tmp2 = call %struct.__neon_int32x4x2_t @llvm.arm.neon.vld2lane.v4i32(i32* %A, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 2)
        %tmp3 = extractvalue %struct.__neon_int32x4x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int32x4x2_t %tmp2, 1
        %tmp5 = add <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

define <4 x float> @vld2laneQf(float* %A, <4 x float>* %B) nounwind {
;CHECK: vld2laneQf:
;CHECK: vld2.32
	%tmp1 = load <4 x float>* %B
	%tmp2 = call %struct.__neon_float32x4x2_t @llvm.arm.neon.vld2lane.v4f32(float* %A, <4 x float> %tmp1, <4 x float> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__neon_float32x4x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_float32x4x2_t %tmp2, 1
        %tmp5 = add <4 x float> %tmp3, %tmp4
	ret <4 x float> %tmp5
}

declare %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2lane.v8i8(i8*, <8 x i8>, <8 x i8>, i32) nounwind readonly
declare %struct.__neon_int16x4x2_t @llvm.arm.neon.vld2lane.v4i16(i8*, <4 x i16>, <4 x i16>, i32) nounwind readonly
declare %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2lane.v2i32(i8*, <2 x i32>, <2 x i32>, i32) nounwind readonly
declare %struct.__neon_float32x2x2_t @llvm.arm.neon.vld2lane.v2f32(i8*, <2 x float>, <2 x float>, i32) nounwind readonly

declare %struct.__neon_int16x8x2_t @llvm.arm.neon.vld2lane.v8i16(i8*, <8 x i16>, <8 x i16>, i32) nounwind readonly
declare %struct.__neon_int32x4x2_t @llvm.arm.neon.vld2lane.v4i32(i8*, <4 x i32>, <4 x i32>, i32) nounwind readonly
declare %struct.__neon_float32x4x2_t @llvm.arm.neon.vld2lane.v4f32(i8*, <4 x float>, <4 x float>, i32) nounwind readonly

%struct.__neon_int8x8x3_t = type { <8 x i8>,  <8 x i8>,  <8 x i8> }
%struct.__neon_int16x4x3_t = type { <4 x i16>, <4 x i16>, <4 x i16> }
%struct.__neon_int32x2x3_t = type { <2 x i32>, <2 x i32>, <2 x i32> }
%struct.__neon_float32x2x3_t = type { <2 x float>, <2 x float>, <2 x float> }

define <8 x i8> @vld3lanei8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK: vld3lanei8:
;CHECK: vld3.8
	%tmp1 = load <8 x i8>* %B
	%tmp2 = call %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3lane.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 2
        %tmp6 = add <8 x i8> %tmp3, %tmp4
        %tmp7 = add <8 x i8> %tmp5, %tmp6
	ret <8 x i8> %tmp7
}

define <4 x i16> @vld3lanei16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK: vld3lanei16:
;CHECK: vld3.16
	%tmp1 = load <4 x i16>* %B
	%tmp2 = call %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3lane.v4i16(i16* %A, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__neon_int16x4x3_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int16x4x3_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int16x4x3_t %tmp2, 2
        %tmp6 = add <4 x i16> %tmp3, %tmp4
        %tmp7 = add <4 x i16> %tmp5, %tmp6
	ret <4 x i16> %tmp7
}

define <2 x i32> @vld3lanei32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK: vld3lanei32:
;CHECK: vld3.32
	%tmp1 = load <2 x i32>* %B
	%tmp2 = call %struct.__neon_int32x2x3_t @llvm.arm.neon.vld3lane.v2i32(i32* %A, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__neon_int32x2x3_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int32x2x3_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int32x2x3_t %tmp2, 2
        %tmp6 = add <2 x i32> %tmp3, %tmp4
        %tmp7 = add <2 x i32> %tmp5, %tmp6
	ret <2 x i32> %tmp7
}

define <2 x float> @vld3lanef(float* %A, <2 x float>* %B) nounwind {
;CHECK: vld3lanef:
;CHECK: vld3.32
	%tmp1 = load <2 x float>* %B
	%tmp2 = call %struct.__neon_float32x2x3_t @llvm.arm.neon.vld3lane.v2f32(float* %A, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__neon_float32x2x3_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_float32x2x3_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_float32x2x3_t %tmp2, 2
        %tmp6 = add <2 x float> %tmp3, %tmp4
        %tmp7 = add <2 x float> %tmp5, %tmp6
	ret <2 x float> %tmp7
}

declare %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3lane.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, i32) nounwind readonly
declare %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3lane.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, i32) nounwind readonly
declare %struct.__neon_int32x2x3_t @llvm.arm.neon.vld3lane.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, i32) nounwind readonly
declare %struct.__neon_float32x2x3_t @llvm.arm.neon.vld3lane.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, i32) nounwind readonly

%struct.__neon_int8x8x4_t = type { <8 x i8>,  <8 x i8>,  <8 x i8>,  <8 x i8> }
%struct.__neon_int16x4x4_t = type { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }
%struct.__neon_int32x2x4_t = type { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }
%struct.__neon_float32x2x4_t = type { <2 x float>, <2 x float>, <2 x float>, <2 x float> }

define <8 x i8> @vld4lanei8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK: vld4lanei8:
;CHECK: vld4.8
	%tmp1 = load <8 x i8>* %B
	%tmp2 = call %struct.__neon_int8x8x4_t @llvm.arm.neon.vld4lane.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 2
        %tmp6 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 3
        %tmp7 = add <8 x i8> %tmp3, %tmp4
        %tmp8 = add <8 x i8> %tmp5, %tmp6
        %tmp9 = add <8 x i8> %tmp7, %tmp8
	ret <8 x i8> %tmp9
}

define <4 x i16> @vld4lanei16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK: vld4lanei16:
;CHECK: vld4.16
	%tmp1 = load <4 x i16>* %B
	%tmp2 = call %struct.__neon_int16x4x4_t @llvm.arm.neon.vld4lane.v4i16(i16* %A, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__neon_int16x4x4_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int16x4x4_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int16x4x4_t %tmp2, 2
        %tmp6 = extractvalue %struct.__neon_int16x4x4_t %tmp2, 3
        %tmp7 = add <4 x i16> %tmp3, %tmp4
        %tmp8 = add <4 x i16> %tmp5, %tmp6
        %tmp9 = add <4 x i16> %tmp7, %tmp8
	ret <4 x i16> %tmp9
}

define <2 x i32> @vld4lanei32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK: vld4lanei32:
;CHECK: vld4.32
	%tmp1 = load <2 x i32>* %B
	%tmp2 = call %struct.__neon_int32x2x4_t @llvm.arm.neon.vld4lane.v2i32(i32* %A, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__neon_int32x2x4_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int32x2x4_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int32x2x4_t %tmp2, 2
        %tmp6 = extractvalue %struct.__neon_int32x2x4_t %tmp2, 3
        %tmp7 = add <2 x i32> %tmp3, %tmp4
        %tmp8 = add <2 x i32> %tmp5, %tmp6
        %tmp9 = add <2 x i32> %tmp7, %tmp8
	ret <2 x i32> %tmp9
}

define <2 x float> @vld4lanef(float* %A, <2 x float>* %B) nounwind {
;CHECK: vld4lanef:
;CHECK: vld4.32
	%tmp1 = load <2 x float>* %B
	%tmp2 = call %struct.__neon_float32x2x4_t @llvm.arm.neon.vld4lane.v2f32(float* %A, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__neon_float32x2x4_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_float32x2x4_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_float32x2x4_t %tmp2, 2
        %tmp6 = extractvalue %struct.__neon_float32x2x4_t %tmp2, 3
        %tmp7 = add <2 x float> %tmp3, %tmp4
        %tmp8 = add <2 x float> %tmp5, %tmp6
        %tmp9 = add <2 x float> %tmp7, %tmp8
	ret <2 x float> %tmp9
}

declare %struct.__neon_int8x8x4_t @llvm.arm.neon.vld4lane.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, i32) nounwind readonly
declare %struct.__neon_int16x4x4_t @llvm.arm.neon.vld4lane.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, i32) nounwind readonly
declare %struct.__neon_int32x2x4_t @llvm.arm.neon.vld4lane.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32) nounwind readonly
declare %struct.__neon_float32x2x4_t @llvm.arm.neon.vld4lane.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, <2 x float>, i32) nounwind readonly
