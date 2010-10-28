; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; XFAIL: *

declare <2 x i32> @llvm.arm.neon.vrecpe.v2i32(<2 x i32>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vrecpe.v4i32(<4 x i32>) nounwind readnone

; CHECK: vrecpe_2xi32
define <2 x i32> @vrecpe_2xi32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vrecpe.u32	d16, d16        @ encoding: [0x20,0x04,0xfb,0xf3]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vrecpe.v2i32(<2 x i32> %tmp1)
	ret <2 x i32> %tmp2
}

; CHECK: vrecpe_4xi32
define <4 x i32> @vrecpe_4xi32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vrecpe.u32	q8, q8          @ encoding: [0x60,0x04,0xfb,0xf3]
	%tmp2 = call <4 x i32> @llvm.arm.neon.vrecpe.v4i32(<4 x i32> %tmp1)
	ret <4 x i32> %tmp2
}

declare <2 x float> @llvm.arm.neon.vrecpe.v2f32(<2 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vrecpe.v4f32(<4 x float>) nounwind readnone

; CHECK: vrecpe_2xfloat
define <2 x float> @vrecpe_2xfloat(<2 x float>* %A) nounwind {
	%tmp1 = load <2 x float>* %A
; CHECK: vrecpe.f32	d16, d16        @ encoding: [0x20,0x05,0xfb,0xf3]
	%tmp2 = call <2 x float> @llvm.arm.neon.vrecpe.v2f32(<2 x float> %tmp1)
	ret <2 x float> %tmp2
}

; CHECK: vrecpe_4xfloat
define <4 x float> @vrecpe_4xfloat(<4 x float>* %A) nounwind {
	%tmp1 = load <4 x float>* %A
; CHECK: vrecpe.f32	q8, q8          @ encoding: [0x60,0x05,0xfb,0xf3]
	%tmp2 = call <4 x float> @llvm.arm.neon.vrecpe.v4f32(<4 x float> %tmp1)
	ret <4 x float> %tmp2
}

declare <2 x float> @llvm.arm.neon.vrecps.v2f32(<2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vrecps.v4f32(<4 x float>, <4 x float>) nounwind readnone

; CHECK: vrecps_2xfloat
define <2 x float> @vrecps_2xfloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
; CHECK: vrecps.f32	d16, d16, d17   @ encoding: [0xb1,0x0f,0x40,0xf2]
	%tmp3 = call <2 x float> @llvm.arm.neon.vrecps.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

; CHECK: vrecps_4xfloat
define <4 x float> @vrecps_4xfloat(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
; CHECK: vrecps.f32	q8, q8, q9      @ encoding: [0xf2,0x0f,0x40,0xf2]
	%tmp3 = call <4 x float> @llvm.arm.neon.vrecps.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x float> %tmp3
}

declare <2 x i32> @llvm.arm.neon.vrsqrte.v2i32(<2 x i32>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vrsqrte.v4i32(<4 x i32>) nounwind readnone

; CHECK: vrsqrte_2xi32
define <2 x i32> @vrsqrte_2xi32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vrsqrte.u32	d16, d16        @ encoding: [0xa0,0x04,0xfb,0xf3]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vrsqrte.v2i32(<2 x i32> %tmp1)
	ret <2 x i32> %tmp2
}

; CHECK: vrsqrte_4xi32
define <4 x i32> @vrsqrte_4xi32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vrsqrte.u32	q8, q8          @ encoding: [0xe0,0x04,0xfb,0xf3]
	%tmp2 = call <4 x i32> @llvm.arm.neon.vrsqrte.v4i32(<4 x i32> %tmp1)
	ret <4 x i32> %tmp2
}

declare <2 x float> @llvm.arm.neon.vrsqrte.v2f32(<2 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vrsqrte.v4f32(<4 x float>) nounwind readnone

; CHECK: vrsqrte_2xfloat
define <2 x float> @vrsqrte_2xfloat(<2 x float>* %A) nounwind {
	%tmp1 = load <2 x float>* %A
; CHECK: vrsqrte.f32	d16, d16        @ encoding: [0xa0,0x05,0xfb,0xf3]
	%tmp2 = call <2 x float> @llvm.arm.neon.vrsqrte.v2f32(<2 x float> %tmp1)
	ret <2 x float> %tmp2
}

; CHECK: vrsqrte_4xfloat
define <4 x float> @vrsqrte_4xfloat(<4 x float>* %A) nounwind {
	%tmp1 = load <4 x float>* %A
; CHECK: vrsqrte.f32	q8, q8          @ encoding: [0xe0,0x05,0xfb,0xf3]
	%tmp2 = call <4 x float> @llvm.arm.neon.vrsqrte.v4f32(<4 x float> %tmp1)
	ret <4 x float> %tmp2
}

declare <2 x float> @llvm.arm.neon.vrsqrts.v2f32(<2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vrsqrts.v4f32(<4 x float>, <4 x float>) nounwind readnone

; CHECK: vrsqrts_2xfloat
define <2 x float> @vrsqrts_2xfloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
; CHECK: vrsqrts.f32	d16, d16, d17   @ encoding: [0xb1,0x0f,0x60,0xf2]
	%tmp3 = call <2 x float> @llvm.arm.neon.vrsqrts.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

; CHECK: vrsqrts_4xfloat
define <4 x float> @vrsqrts_4xfloat(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
; CHECK: vrsqrts.f32	q8, q8, q9      @ encoding: [0xf2,0x0f,0x60,0xf2]
	%tmp3 = call <4 x float> @llvm.arm.neon.vrsqrts.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x float> %tmp3
}
