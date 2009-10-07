; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vmuli8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vmuli8:
;CHECK: vmul.i8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = mul <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vmuli16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vmuli16:
;CHECK: vmul.i16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = mul <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vmuli32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vmuli32:
;CHECK: vmul.i32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = mul <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <2 x float> @vmulf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vmulf32:
;CHECK: vmul.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = mul <2 x float> %tmp1, %tmp2
	ret <2 x float> %tmp3
}

define <8 x i8> @vmulp8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vmulp8:
;CHECK: vmul.p8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vmulp.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <16 x i8> @vmulQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vmulQi8:
;CHECK: vmul.i8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = mul <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vmulQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vmulQi16:
;CHECK: vmul.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = mul <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vmulQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vmulQi32:
;CHECK: vmul.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = mul <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <4 x float> @vmulQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK: vmulQf32:
;CHECK: vmul.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = mul <4 x float> %tmp1, %tmp2
	ret <4 x float> %tmp3
}

define <16 x i8> @vmulQp8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vmulQp8:
;CHECK: vmul.p8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vmulp.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vmulp.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8>  @llvm.arm.neon.vmulp.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
