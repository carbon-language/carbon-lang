; RUN: llvm-as < %s | llc -march=arm -mattr=+neon | FileCheck %s

%struct.__builtin_neon_v8qi2 = type { <8 x i8>,  <8 x i8> }
%struct.__builtin_neon_v4hi2 = type { <4 x i16>, <4 x i16> }
%struct.__builtin_neon_v2si2 = type { <2 x i32>, <2 x i32> }
%struct.__builtin_neon_v2sf2 = type { <2 x float>, <2 x float> }

%struct.__builtin_neon_v16qi2 = type { <16 x i8>, <16 x i8> }
%struct.__builtin_neon_v8hi2  = type { <8 x i16>, <8 x i16> }
%struct.__builtin_neon_v4si2  = type { <4 x i32>, <4 x i32> }
%struct.__builtin_neon_v4sf2  = type { <4 x float>, <4 x float> }

define <8 x i8> @vuzpi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vuzpi8:
;CHECK: vuzp.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call %struct.__builtin_neon_v8qi2 @llvm.arm.neon.vuzp.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
        %tmp4 = extractvalue %struct.__builtin_neon_v8qi2 %tmp3, 0
        %tmp5 = extractvalue %struct.__builtin_neon_v8qi2 %tmp3, 1
        %tmp6 = add <8 x i8> %tmp4, %tmp5
	ret <8 x i8> %tmp6
}

define <4 x i16> @vuzpi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vuzpi16:
;CHECK: vuzp.16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call %struct.__builtin_neon_v4hi2 @llvm.arm.neon.vuzp.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
        %tmp4 = extractvalue %struct.__builtin_neon_v4hi2 %tmp3, 0
        %tmp5 = extractvalue %struct.__builtin_neon_v4hi2 %tmp3, 1
        %tmp6 = add <4 x i16> %tmp4, %tmp5
	ret <4 x i16> %tmp6
}

define <2 x i32> @vuzpi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vuzpi32:
;CHECK: vuzp.32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call %struct.__builtin_neon_v2si2 @llvm.arm.neon.vuzp.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
        %tmp4 = extractvalue %struct.__builtin_neon_v2si2 %tmp3, 0
        %tmp5 = extractvalue %struct.__builtin_neon_v2si2 %tmp3, 1
        %tmp6 = add <2 x i32> %tmp4, %tmp5
	ret <2 x i32> %tmp6
}

define <2 x float> @vuzpf(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vuzpf:
;CHECK: vuzp.32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = call %struct.__builtin_neon_v2sf2 @llvm.arm.neon.vuzp.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
        %tmp4 = extractvalue %struct.__builtin_neon_v2sf2 %tmp3, 0
        %tmp5 = extractvalue %struct.__builtin_neon_v2sf2 %tmp3, 1
        %tmp6 = add <2 x float> %tmp4, %tmp5
	ret <2 x float> %tmp6
}

define <16 x i8> @vuzpQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vuzpQi8:
;CHECK: vuzp.8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = call %struct.__builtin_neon_v16qi2 @llvm.arm.neon.vuzp.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
        %tmp4 = extractvalue %struct.__builtin_neon_v16qi2 %tmp3, 0
        %tmp5 = extractvalue %struct.__builtin_neon_v16qi2 %tmp3, 1
        %tmp6 = add <16 x i8> %tmp4, %tmp5
	ret <16 x i8> %tmp6
}

define <8 x i16> @vuzpQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vuzpQi16:
;CHECK: vuzp.16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = call %struct.__builtin_neon_v8hi2 @llvm.arm.neon.vuzp.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
        %tmp4 = extractvalue %struct.__builtin_neon_v8hi2 %tmp3, 0
        %tmp5 = extractvalue %struct.__builtin_neon_v8hi2 %tmp3, 1
        %tmp6 = add <8 x i16> %tmp4, %tmp5
	ret <8 x i16> %tmp6
}

define <4 x i32> @vuzpQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vuzpQi32:
;CHECK: vuzp.32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = call %struct.__builtin_neon_v4si2 @llvm.arm.neon.vuzp.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
        %tmp4 = extractvalue %struct.__builtin_neon_v4si2 %tmp3, 0
        %tmp5 = extractvalue %struct.__builtin_neon_v4si2 %tmp3, 1
        %tmp6 = add <4 x i32> %tmp4, %tmp5
	ret <4 x i32> %tmp6
}

define <4 x float> @vuzpQf(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK: vuzpQf:
;CHECK: vuzp.32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = call %struct.__builtin_neon_v4sf2 @llvm.arm.neon.vuzp.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
        %tmp4 = extractvalue %struct.__builtin_neon_v4sf2 %tmp3, 0
        %tmp5 = extractvalue %struct.__builtin_neon_v4sf2 %tmp3, 1
        %tmp6 = add <4 x float> %tmp4, %tmp5
	ret <4 x float> %tmp6
}

declare %struct.__builtin_neon_v8qi2 @llvm.arm.neon.vuzp.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare %struct.__builtin_neon_v4hi2 @llvm.arm.neon.vuzp.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare %struct.__builtin_neon_v2si2 @llvm.arm.neon.vuzp.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare %struct.__builtin_neon_v2sf2 @llvm.arm.neon.vuzp.v2f32(<2 x float>, <2 x float>) nounwind readnone

declare %struct.__builtin_neon_v16qi2 @llvm.arm.neon.vuzp.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare %struct.__builtin_neon_v8hi2 @llvm.arm.neon.vuzp.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare %struct.__builtin_neon_v4si2 @llvm.arm.neon.vuzp.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare %struct.__builtin_neon_v4sf2 @llvm.arm.neon.vuzp.v4f32(<4 x float>, <4 x float>) nounwind readnone
