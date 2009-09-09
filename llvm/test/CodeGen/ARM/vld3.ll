; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

%struct.__builtin_neon_v8qi3 = type { <8 x i8>,  <8 x i8>,  <8 x i8> }
%struct.__builtin_neon_v4hi3 = type { <4 x i16>, <4 x i16>, <4 x i16> }
%struct.__builtin_neon_v2si3 = type { <2 x i32>, <2 x i32>, <2 x i32> }
%struct.__builtin_neon_v2sf3 = type { <2 x float>, <2 x float>, <2 x float> }

define <8 x i8> @vld3i8(i8* %A) nounwind {
;CHECK: vld3i8:
;CHECK: vld3.8
	%tmp1 = call %struct.__builtin_neon_v8qi3 @llvm.arm.neon.vld3.v8i8(i8* %A)
        %tmp2 = extractvalue %struct.__builtin_neon_v8qi3 %tmp1, 0
        %tmp3 = extractvalue %struct.__builtin_neon_v8qi3 %tmp1, 2
        %tmp4 = add <8 x i8> %tmp2, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @vld3i16(i16* %A) nounwind {
;CHECK: vld3i16:
;CHECK: vld3.16
	%tmp1 = call %struct.__builtin_neon_v4hi3 @llvm.arm.neon.vld3.v4i16(i16* %A)
        %tmp2 = extractvalue %struct.__builtin_neon_v4hi3 %tmp1, 0
        %tmp3 = extractvalue %struct.__builtin_neon_v4hi3 %tmp1, 2
        %tmp4 = add <4 x i16> %tmp2, %tmp3
	ret <4 x i16> %tmp4
}

define <2 x i32> @vld3i32(i32* %A) nounwind {
;CHECK: vld3i32:
;CHECK: vld3.32
	%tmp1 = call %struct.__builtin_neon_v2si3 @llvm.arm.neon.vld3.v2i32(i32* %A)
        %tmp2 = extractvalue %struct.__builtin_neon_v2si3 %tmp1, 0
        %tmp3 = extractvalue %struct.__builtin_neon_v2si3 %tmp1, 2
        %tmp4 = add <2 x i32> %tmp2, %tmp3
	ret <2 x i32> %tmp4
}

define <2 x float> @vld3f(float* %A) nounwind {
;CHECK: vld3f:
;CHECK: vld3.32
	%tmp1 = call %struct.__builtin_neon_v2sf3 @llvm.arm.neon.vld3.v2f32(float* %A)
        %tmp2 = extractvalue %struct.__builtin_neon_v2sf3 %tmp1, 0
        %tmp3 = extractvalue %struct.__builtin_neon_v2sf3 %tmp1, 2
        %tmp4 = add <2 x float> %tmp2, %tmp3
	ret <2 x float> %tmp4
}

declare %struct.__builtin_neon_v8qi3 @llvm.arm.neon.vld3.v8i8(i8*) nounwind readonly
declare %struct.__builtin_neon_v4hi3 @llvm.arm.neon.vld3.v4i16(i8*) nounwind readonly
declare %struct.__builtin_neon_v2si3 @llvm.arm.neon.vld3.v2i32(i8*) nounwind readonly
declare %struct.__builtin_neon_v2sf3 @llvm.arm.neon.vld3.v2f32(i8*) nounwind readonly
