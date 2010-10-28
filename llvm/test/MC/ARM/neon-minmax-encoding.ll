; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; XFAIL: *

declare <8 x i8>  @llvm.arm.neon.vmins.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vmins.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vmins.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vmins_8xi8
define <8 x i8> @vmins_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vmin.s8	d16, d16, d17           @ encoding: [0xb1,0x06,0x40,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vmins.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vmins_4xi16
define <4 x i16> @vmins_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vmin.s16	d16, d16, d17   @ encoding: [0xb1,0x06,0x50,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vmins.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vmins_2xi32
define <2 x i32> @vmins_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vmin.s32	d16, d16, d17   @ encoding: [0xb1,0x06,0x60,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vmins.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vminu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vminu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vminu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vminu_8xi8
define <8 x i8> @vminu_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vmin.u8	d16, d16, d17           @ encoding: [0xb1,0x06,0x40,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vminu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vminu_4xi16
define <4 x i16> @vminu_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vmin.u16	d16, d16, d17   @ encoding: [0xb1,0x06,0x50,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vminu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vminu_2xi32
define <2 x i32> @vminu_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vmin.u32	d16, d16, d17   @ encoding: [0xb1,0x06,0x60,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vminu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <2 x float> @llvm.arm.neon.vmins.v2f32(<2 x float>, <2 x float>) nounwind readnone

; CHECK: vmin_2xfloat
define <2 x float> @vmin_2xfloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
; CHECK: vmin.f32	d16, d16, d17   @ encoding: [0xa1,0x0f,0x60,0xf2]
	%tmp3 = call <2 x float> @llvm.arm.neon.vmins.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

declare <16 x i8> @llvm.arm.neon.vmins.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vmins.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmins.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

; CHECK: vmins_16xi8
define <16 x i8> @vmins_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vmin.s8	q8, q8, q9              @ encoding: [0xf2,0x06,0x40,0xf2]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vmins.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

; CHECK: vmins_8xi16
define <8 x i16> @vmins_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vmin.s16	q8, q8, q9      @ encoding: [0xf2,0x06,0x50,0xf2]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vmins.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vmins_4xi32
define <4 x i32> @vmins_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vmin.s32	q8, q8, q9      @ encoding: [0xf2,0x06,0x60,0xf2]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vmins.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <16 x i8> @llvm.arm.neon.vminu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vminu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vminu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

; CHECK: vminu_16xi8
define <16 x i8> @vminu_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vmin.u8	q8, q8, q9              @ encoding: [0xf2,0x06,0x40,0xf3]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vminu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

; CHECK: vminu_8xi16
define <8 x i16> @vminu_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vmin.u16	q8, q8, q9      @ encoding: [0xf2,0x06,0x50,0xf3]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vminu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vminu_4xi32
define <4 x i32> @vminu_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vmin.u32	q8, q8, q9      @ encoding: [0xf2,0x06,0x60,0xf3]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vminu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <4 x float> @llvm.arm.neon.vmins.v4f32(<4 x float>, <4 x float>) nounwind readnone

; CHECK: vmin_4xfloat
define <4 x float> @vmin_4xfloat(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
; CHECK: vmin.f32	q8, q8, q9      @ encoding: [0xe2,0x0f,0x60,0xf2]
	%tmp3 = call <4 x float> @llvm.arm.neon.vmins.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x float> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vmaxs.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vmaxs.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vmaxs.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vmaxs_8xi8
define <8 x i8> @vmaxs_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vmax.s8	d16, d16, d17           @ encoding: [0xa1,0x06,0x40,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vmaxs.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vmaxs_4xi16
define <4 x i16> @vmaxs_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vmax.s16	d16, d16, d17   @ encoding: [0xa1,0x06,0x50,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vmaxs.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vmaxs_2xi32
define <2 x i32> @vmaxs_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vmax.s32	d16, d16, d17   @ encoding: [0xa1,0x06,0x60,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vmaxs.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vmaxu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vmaxu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vmaxu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vmaxu_8xi8
define <8 x i8> @vmaxu_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vmax.u8	d16, d16, d17           @ encoding: [0xa1,0x06,0x40,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vmaxu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vmaxu_4xi16
define <4 x i16> @vmaxu_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vmax.u16	d16, d16, d17   @ encoding: [0xa1,0x06,0x50,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vmaxu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vmaxu_2xi32
define <2 x i32> @vmaxu_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vmax.u32	d16, d16, d17   @ encoding: [0xa1,0x06,0x60,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vmaxu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <2 x float> @llvm.arm.neon.vmaxs.v2f32(<2 x float>, <2 x float>) nounwind readnone

; CHECK: vmax_2xfloat
define <2 x float> @vmax_2xfloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
; CHECK: vmax.f32	d16, d16, d17   @ encoding: [0xa1,0x0f,0x40,0xf2]
	%tmp3 = call <2 x float> @llvm.arm.neon.vmaxs.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

declare <16 x i8> @llvm.arm.neon.vmaxs.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vmaxs.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmaxs.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

; CHECK: vmaxs_16xi8
define <16 x i8> @vmaxs_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vmax.s8	q8, q8, q9              @ encoding: [0xe2,0x06,0x40,0xf2]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vmaxs.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

; CHECK: vmaxs_8xi16
define <8 x i16> @vmaxs_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vmax.s16	q8, q8, q9      @ encoding: [0xe2,0x06,0x50,0xf2]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vmaxs.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vmaxs_4xi32
define <4 x i32> @vmaxs_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vmax.s32	q8, q8, q9      @ encoding: [0xe2,0x06,0x60,0xf2]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vmaxs.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <16 x i8> @llvm.arm.neon.vmaxu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vmaxu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmaxu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

; CHECK: vmaxu_16xi8
define <16 x i8> @vmaxu_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vmax.u8	q8, q8, q9              @ encoding: [0xe2,0x06,0x40,0xf3]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vmaxu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

; CHECK: vmaxu_8xi16
define <8 x i16> @vmaxu_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vmax.u16	q8, q8, q9      @ encoding: [0xe2,0x06,0x50,0xf3]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vmaxu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vmaxu_4xi32
define <4 x i32> @vmaxu_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vmax.u32	q8, q8, q9      @ encoding: [0xe2,0x06,0x60,0xf3]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vmaxu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <4 x float> @llvm.arm.neon.vmaxs.v4f32(<4 x float>, <4 x float>) nounwind readnone

; CHECK: vmax_4xfloat
define <4 x float> @vmax_4xfloat(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
; CHECK: vmax.f32	q8, q8, q9      @ encoding: [0xe2,0x0f,0x40,0xf2]
	%tmp3 = call <4 x float> @llvm.arm.neon.vmaxs.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x float> %tmp3
}
