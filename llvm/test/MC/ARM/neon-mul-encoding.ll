; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; XFAIL: *

; CHECK: vmul_8xi8
define <8 x i8> @vmul_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vmul.i8	d16, d16, d17           @ encoding: [0xb1,0x09,0x40,0xf2]
	%tmp3 = mul <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

; CHECK: vmul_4xi16
define <4 x i16> @vmul_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vmul.i16	d16, d16, d17   @ encoding: [0xb1,0x09,0x50,0xf2]
	%tmp3 = mul <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

; CHECK: vmul_2xi32
define <2 x i32> @vmul_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vmul.i32	d16, d16, d17   @ encoding: [0xb1,0x09,0x60,0xf2]
	%tmp3 = mul <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

; CHECK: vmul_2xfloat
define <2 x float> @vmul_2xfloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
; CHECK: vmul.f32	d16, d16, d17   @ encoding: [0xb1,0x0d,0x40,0xf3]
	%tmp3 = fmul <2 x float> %tmp1, %tmp2
	ret <2 x float> %tmp3
}

; CHECK: vmul_16xi8
define <16 x i8> @vmul_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vmul.i8	q8, q8, q9              @ encoding: [0xf2,0x09,0x40,0xf2]
	%tmp3 = mul <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

; CHECK: vmul_8xi16
define <8 x i16> @vmul_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vmul.i16	q8, q8, q9      @ encoding: [0xf2,0x09,0x50,0xf2]
	%tmp3 = mul <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

; CHECK: vmul_4xi32
define <4 x i32> @vmul_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vmul.i32	q8, q8, q9      @ encoding: [0xf2,0x09,0x60,0xf2]
	%tmp3 = mul <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

; CHECK: vmul_4xfloat
define <4 x float> @vmul_4xfloat(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
; CHECK: vmul.f32	q8, q8, q9      @ encoding: [0xf2,0x0d,0x40,0xf3]
	%tmp3 = fmul <4 x float> %tmp1, %tmp2
	ret <4 x float> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vmulp.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8>  @llvm.arm.neon.vmulp.v16i8(<16 x i8>, <16 x i8>) nounwind readnone

; CHECK: vmulp_8xi8
define <8 x i8> @vmulp_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vmul.p8	d16, d16, d17           @ encoding: [0xb1,0x09,0x40,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vmulp.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vmulp_16xi8
define <16 x i8> @vmulp_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vmul.p8	q8, q8, q9              @ encoding: [0xf2,0x09,0x40,0xf3]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vmulp.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

declare <4 x i16> @llvm.arm.neon.vqdmulh.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqdmulh.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vqdmulh_4xi16
define <4 x i16> @vqdmulh_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vqdmulh.s16	d16, d16, d17   @ encoding: [0xa1,0x0b,0x50,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqdmulh.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vqdmulh_2xi32
define <2 x i32> @vqdmulh_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vqdmulh.s32	d16, d16, d17   @ encoding: [0xa1,0x0b,0x60,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqdmulh.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i16> @llvm.arm.neon.vqdmulh.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqdmulh.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

; CHECK: vqdmulh_8xi16
define <8 x i16> @vqdmulh_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vqdmulh.s16	q8, q8, q9      @ encoding: [0xe2,0x0b,0x50,0xf2]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqdmulh.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vqdmulh_4xi32
define <4 x i32> @vqdmulh_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vqdmulh.s32	q8, q8, q9      @ encoding: [0xe2,0x0b,0x60,0xf2]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqdmulh.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16>, <4 x i16>) nounwind readnone

; CHECK: vqrdmulh_4xi16
define <4 x i16> @vqrdmulh_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vqrdmulh.s16	d16, d16, d17   @ encoding: [0xa1,0x0b,0x50,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vqrdmulh_2xi32
define <2 x i32> @vqrdmulh_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vqrdmulh.s32	d16, d16, d17   @ encoding: [0xa1,0x0b,0x60,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

; CHECK: vqrdmulh_8xi16
define <8 x i16> @vqrdmulh_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vqrdmulh.s16	q8, q8, q9      @ encoding: [0xe2,0x0b,0x50,0xf3]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vqrdmulh_4xi32
define <4 x i32> @vqrdmulh_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vqrdmulh.s32	q8, q8, q9      @ encoding: [0xe2,0x0b,0x60,0xf3]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

; CHECK: vmulls_8xi16
define <8 x i16> @vmulls_8xi16(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = sext <8 x i8> %tmp1 to <8 x i16>
	%tmp4 = sext <8 x i8> %tmp2 to <8 x i16>
; CHECK: vmull.s8	q8, d16, d17    @ encoding: [0xa1,0x0c,0xc0,0xf2]
	%tmp5 = mul <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

; CHECK: vmulls_4xi16
define <4 x i32> @vmulls_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = sext <4 x i16> %tmp1 to <4 x i32>
	%tmp4 = sext <4 x i16> %tmp2 to <4 x i32>
; CHECK: vmull.s16	q8, d16, d17    @ encoding: [0xa1,0x0c,0xd0,0xf2]
	%tmp5 = mul <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

; CHECK: vmulls_2xi32
define <2 x i64> @vmulls_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = sext <2 x i32> %tmp1 to <2 x i64>
	%tmp4 = sext <2 x i32> %tmp2 to <2 x i64>
; CHECK: vmull.s32	q8, d16, d17    @ encoding: [0xa1,0x0c,0xe0,0xf2]
	%tmp5 = mul <2 x i64> %tmp3, %tmp4
	ret <2 x i64> %tmp5
}

; CHECK: vmullu_8xi8
define <8 x i16> @vmullu_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = zext <8 x i8> %tmp1 to <8 x i16>
	%tmp4 = zext <8 x i8> %tmp2 to <8 x i16>
; CHECK: vmull.u8	q8, d16, d17    @ encoding: [0xa1,0x0c,0xc0,0xf3]
	%tmp5 = mul <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

; CHECK: vmullu_4xi16
define <4 x i32> @vmullu_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = zext <4 x i16> %tmp1 to <4 x i32>
	%tmp4 = zext <4 x i16> %tmp2 to <4 x i32>
; CHECK: vmull.u16	q8, d16, d17    @ encoding: [0xa1,0x0c,0xd0,0xf3]
	%tmp5 = mul <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

; CHECK: vmullu_2xi32
define <2 x i64> @vmullu_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = zext <2 x i32> %tmp1 to <2 x i64>
	%tmp4 = zext <2 x i32> %tmp2 to <2 x i64>
; CHECK: vmull.u32	q8, d16, d17    @ encoding: [0xa1,0x0c,0xe0,0xf3]
	%tmp5 = mul <2 x i64> %tmp3, %tmp4
	ret <2 x i64> %tmp5
}

declare <8 x i16>  @llvm.arm.neon.vmullp.v8i16(<8 x i8>, <8 x i8>) nounwind readnone

; CHECK: vmullp_8xi8
define <8 x i16> @vmullp_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vmull.p8	q8, d16, d17    @ encoding: [0xa1,0x0e,0xc0,0xf2]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vmullp.v8i16(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

declare <4 x i32>  @llvm.arm.neon.vqdmull.v4i32(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64>  @llvm.arm.neon.vqdmull.v2i64(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vqdmull_4xi16
define <4 x i32> @vqdmull_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vqdmull.s16	q8, d16, d17    @ encoding: [0xa1,0x0d,0xd0,0xf2]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

; CHECK: vqdmull_2xi32
define <2 x i64> @vqdmull_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vqdmull.s32	q8, d16, d17    @ encoding: [0xa1,0x0d,0xe0,0xf2]
	%tmp3 = call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i64> %tmp3
}
