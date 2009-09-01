; RUN: llvm-as < %s | llc -march=arm -mattr=+neon | FileCheck %s

%struct.__builtin_neon_v8qi2 = type { <8 x i8>,  <8 x i8> }
%struct.__builtin_neon_v4hi2 = type { <4 x i16>, <4 x i16> }
%struct.__builtin_neon_v2si2 = type { <2 x i32>, <2 x i32> }
%struct.__builtin_neon_v2sf2 = type { <2 x float>, <2 x float> }

define <8 x i8> @vld2lanei8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK: vld2lanei8:
;CHECK: vld2.8
	%tmp1 = load <8 x i8>* %B
	%tmp2 = call %struct.__builtin_neon_v8qi2 @llvm.arm.neon.vld2lane.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__builtin_neon_v8qi2 %tmp2, 0
        %tmp4 = extractvalue %struct.__builtin_neon_v8qi2 %tmp2, 1
        %tmp5 = add <8 x i8> %tmp3, %tmp4
	ret <8 x i8> %tmp5
}

define <4 x i16> @vld2lanei16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK: vld2lanei16:
;CHECK: vld2.16
	%tmp1 = load <4 x i16>* %B
	%tmp2 = call %struct.__builtin_neon_v4hi2 @llvm.arm.neon.vld2lane.v4i16(i16* %A, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__builtin_neon_v4hi2 %tmp2, 0
        %tmp4 = extractvalue %struct.__builtin_neon_v4hi2 %tmp2, 1
        %tmp5 = add <4 x i16> %tmp3, %tmp4
	ret <4 x i16> %tmp5
}

define <2 x i32> @vld2lanei32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK: vld2lanei32:
;CHECK: vld2.32
	%tmp1 = load <2 x i32>* %B
	%tmp2 = call %struct.__builtin_neon_v2si2 @llvm.arm.neon.vld2lane.v2i32(i32* %A, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__builtin_neon_v2si2 %tmp2, 0
        %tmp4 = extractvalue %struct.__builtin_neon_v2si2 %tmp2, 1
        %tmp5 = add <2 x i32> %tmp3, %tmp4
	ret <2 x i32> %tmp5
}

define <2 x float> @vld2lanef(float* %A, <2 x float>* %B) nounwind {
;CHECK: vld2lanef:
;CHECK: vld2.32
	%tmp1 = load <2 x float>* %B
	%tmp2 = call %struct.__builtin_neon_v2sf2 @llvm.arm.neon.vld2lane.v2f32(float* %A, <2 x float> %tmp1, <2 x float> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__builtin_neon_v2sf2 %tmp2, 0
        %tmp4 = extractvalue %struct.__builtin_neon_v2sf2 %tmp2, 1
        %tmp5 = add <2 x float> %tmp3, %tmp4
	ret <2 x float> %tmp5
}

declare %struct.__builtin_neon_v8qi2 @llvm.arm.neon.vld2lane.v8i8(i8*) nounwind readonly
declare %struct.__builtin_neon_v4hi2 @llvm.arm.neon.vld2lane.v4i16(i8*) nounwind readonly
declare %struct.__builtin_neon_v2si2 @llvm.arm.neon.vld2lane.v2i32(i8*) nounwind readonly
declare %struct.__builtin_neon_v2sf2 @llvm.arm.neon.vld2lane.v2f32(i8*) nounwind readonly

%struct.__builtin_neon_v8qi3 = type { <8 x i8>,  <8 x i8>,  <8 x i8> }
%struct.__builtin_neon_v4hi3 = type { <4 x i16>, <4 x i16>, <4 x i16> }
%struct.__builtin_neon_v2si3 = type { <2 x i32>, <2 x i32>, <2 x i32> }
%struct.__builtin_neon_v2sf3 = type { <2 x float>, <2 x float>, <2 x float> }

define <8 x i8> @vld3lanei8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK: vld3lanei8:
;CHECK: vld3.8
	%tmp1 = load <8 x i8>* %B
	%tmp2 = call %struct.__builtin_neon_v8qi3 @llvm.arm.neon.vld3lane.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__builtin_neon_v8qi3 %tmp2, 0
        %tmp4 = extractvalue %struct.__builtin_neon_v8qi3 %tmp2, 1
        %tmp5 = extractvalue %struct.__builtin_neon_v8qi3 %tmp2, 2
        %tmp6 = add <8 x i8> %tmp3, %tmp4
        %tmp7 = add <8 x i8> %tmp5, %tmp6
	ret <8 x i8> %tmp7
}

define <4 x i16> @vld3lanei16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK: vld3lanei16:
;CHECK: vld3.16
	%tmp1 = load <4 x i16>* %B
	%tmp2 = call %struct.__builtin_neon_v4hi3 @llvm.arm.neon.vld3lane.v4i16(i16* %A, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__builtin_neon_v4hi3 %tmp2, 0
        %tmp4 = extractvalue %struct.__builtin_neon_v4hi3 %tmp2, 1
        %tmp5 = extractvalue %struct.__builtin_neon_v4hi3 %tmp2, 2
        %tmp6 = add <4 x i16> %tmp3, %tmp4
        %tmp7 = add <4 x i16> %tmp5, %tmp6
	ret <4 x i16> %tmp7
}

define <2 x i32> @vld3lanei32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK: vld3lanei32:
;CHECK: vld3.32
	%tmp1 = load <2 x i32>* %B
	%tmp2 = call %struct.__builtin_neon_v2si3 @llvm.arm.neon.vld3lane.v2i32(i32* %A, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__builtin_neon_v2si3 %tmp2, 0
        %tmp4 = extractvalue %struct.__builtin_neon_v2si3 %tmp2, 1
        %tmp5 = extractvalue %struct.__builtin_neon_v2si3 %tmp2, 2
        %tmp6 = add <2 x i32> %tmp3, %tmp4
        %tmp7 = add <2 x i32> %tmp5, %tmp6
	ret <2 x i32> %tmp7
}

define <2 x float> @vld3lanef(float* %A, <2 x float>* %B) nounwind {
;CHECK: vld3lanef:
;CHECK: vld3.32
	%tmp1 = load <2 x float>* %B
	%tmp2 = call %struct.__builtin_neon_v2sf3 @llvm.arm.neon.vld3lane.v2f32(float* %A, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__builtin_neon_v2sf3 %tmp2, 0
        %tmp4 = extractvalue %struct.__builtin_neon_v2sf3 %tmp2, 1
        %tmp5 = extractvalue %struct.__builtin_neon_v2sf3 %tmp2, 2
        %tmp6 = add <2 x float> %tmp3, %tmp4
        %tmp7 = add <2 x float> %tmp5, %tmp6
	ret <2 x float> %tmp7
}

declare %struct.__builtin_neon_v8qi3 @llvm.arm.neon.vld3lane.v8i8(i8*) nounwind readonly
declare %struct.__builtin_neon_v4hi3 @llvm.arm.neon.vld3lane.v4i16(i8*) nounwind readonly
declare %struct.__builtin_neon_v2si3 @llvm.arm.neon.vld3lane.v2i32(i8*) nounwind readonly
declare %struct.__builtin_neon_v2sf3 @llvm.arm.neon.vld3lane.v2f32(i8*) nounwind readonly

%struct.__builtin_neon_v8qi4 = type { <8 x i8>,  <8 x i8>,  <8 x i8>,  <8 x i8> }
%struct.__builtin_neon_v4hi4 = type { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }
%struct.__builtin_neon_v2si4 = type { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }
%struct.__builtin_neon_v2sf4 = type { <2 x float>, <2 x float>, <2 x float>, <2 x float> }

define <8 x i8> @vld4lanei8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK: vld4lanei8:
;CHECK: vld4.8
	%tmp1 = load <8 x i8>* %B
	%tmp2 = call %struct.__builtin_neon_v8qi4 @llvm.arm.neon.vld4lane.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__builtin_neon_v8qi4 %tmp2, 0
        %tmp4 = extractvalue %struct.__builtin_neon_v8qi4 %tmp2, 1
        %tmp5 = extractvalue %struct.__builtin_neon_v8qi4 %tmp2, 2
        %tmp6 = extractvalue %struct.__builtin_neon_v8qi4 %tmp2, 3
        %tmp7 = add <8 x i8> %tmp3, %tmp4
        %tmp8 = add <8 x i8> %tmp5, %tmp6
        %tmp9 = add <8 x i8> %tmp7, %tmp8
	ret <8 x i8> %tmp9
}

define <4 x i16> @vld4lanei16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK: vld4lanei16:
;CHECK: vld4.16
	%tmp1 = load <4 x i16>* %B
	%tmp2 = call %struct.__builtin_neon_v4hi4 @llvm.arm.neon.vld4lane.v4i16(i16* %A, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__builtin_neon_v4hi4 %tmp2, 0
        %tmp4 = extractvalue %struct.__builtin_neon_v4hi4 %tmp2, 1
        %tmp5 = extractvalue %struct.__builtin_neon_v4hi4 %tmp2, 2
        %tmp6 = extractvalue %struct.__builtin_neon_v4hi4 %tmp2, 3
        %tmp7 = add <4 x i16> %tmp3, %tmp4
        %tmp8 = add <4 x i16> %tmp5, %tmp6
        %tmp9 = add <4 x i16> %tmp7, %tmp8
	ret <4 x i16> %tmp9
}

define <2 x i32> @vld4lanei32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK: vld4lanei32:
;CHECK: vld4.32
	%tmp1 = load <2 x i32>* %B
	%tmp2 = call %struct.__builtin_neon_v2si4 @llvm.arm.neon.vld4lane.v2i32(i32* %A, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__builtin_neon_v2si4 %tmp2, 0
        %tmp4 = extractvalue %struct.__builtin_neon_v2si4 %tmp2, 1
        %tmp5 = extractvalue %struct.__builtin_neon_v2si4 %tmp2, 2
        %tmp6 = extractvalue %struct.__builtin_neon_v2si4 %tmp2, 3
        %tmp7 = add <2 x i32> %tmp3, %tmp4
        %tmp8 = add <2 x i32> %tmp5, %tmp6
        %tmp9 = add <2 x i32> %tmp7, %tmp8
	ret <2 x i32> %tmp9
}

define <2 x float> @vld4lanef(float* %A, <2 x float>* %B) nounwind {
;CHECK: vld4lanef:
;CHECK: vld4.32
	%tmp1 = load <2 x float>* %B
	%tmp2 = call %struct.__builtin_neon_v2sf4 @llvm.arm.neon.vld4lane.v2f32(float* %A, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, i32 1)
        %tmp3 = extractvalue %struct.__builtin_neon_v2sf4 %tmp2, 0
        %tmp4 = extractvalue %struct.__builtin_neon_v2sf4 %tmp2, 1
        %tmp5 = extractvalue %struct.__builtin_neon_v2sf4 %tmp2, 2
        %tmp6 = extractvalue %struct.__builtin_neon_v2sf4 %tmp2, 3
        %tmp7 = add <2 x float> %tmp3, %tmp4
        %tmp8 = add <2 x float> %tmp5, %tmp6
        %tmp9 = add <2 x float> %tmp7, %tmp8
	ret <2 x float> %tmp9
}

declare %struct.__builtin_neon_v8qi4 @llvm.arm.neon.vld4lane.v8i8(i8*) nounwind readonly
declare %struct.__builtin_neon_v4hi4 @llvm.arm.neon.vld4lane.v4i16(i8*) nounwind readonly
declare %struct.__builtin_neon_v2si4 @llvm.arm.neon.vld4lane.v2i32(i8*) nounwind readonly
declare %struct.__builtin_neon_v2sf4 @llvm.arm.neon.vld4lane.v2f32(i8*) nounwind readonly
