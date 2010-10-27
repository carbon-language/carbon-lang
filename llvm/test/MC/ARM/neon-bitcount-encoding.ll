; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

define <8 x i8> @vcnt8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vcnt.8	d16, d16                @ encoding: [0x20,0x05,0xf0,0xf3]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vcnt.v8i8(<8 x i8> %tmp1)
	ret <8 x i8> %tmp2
}

define <16 x i8> @vcntQ8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vcnt.8	q8, q8                  @ encoding: [0x60,0x05,0xf0,0xf3]
	%tmp2 = call <16 x i8> @llvm.arm.neon.vcnt.v16i8(<16 x i8> %tmp1)
	ret <16 x i8> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vcnt.v8i8(<8 x i8>) nounwind readnone
declare <16 x i8> @llvm.arm.neon.vcnt.v16i8(<16 x i8>) nounwind readnone

define <8 x i8> @vclz8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vclz.i8	d16, d16                @ encoding: [0xa0,0x04,0xf0,0xf3]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vclz.v8i8(<8 x i8> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vclz16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vclz.i16	d16, d16        @ encoding: [0xa0,0x04,0xf4,0xf3]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vclz.v4i16(<4 x i16> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vclz32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vclz.i32	d16, d16        @ encoding: [0xa0,0x04,0xf8,0xf3]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vclz.v2i32(<2 x i32> %tmp1)
	ret <2 x i32> %tmp2
}

define <16 x i8> @vclzQ8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vclz.i8	q8, q8                  @ encoding: [0xe0,0x04,0xf0,0xf3]
	%tmp2 = call <16 x i8> @llvm.arm.neon.vclz.v16i8(<16 x i8> %tmp1)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vclzQ16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vclz.i16	q8, q8          @ encoding: [0xe0,0x04,0xf4,0xf3]
	%tmp2 = call <8 x i16> @llvm.arm.neon.vclz.v8i16(<8 x i16> %tmp1)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vclzQ32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vclz.i32	q8, q8          @ encoding: [0xe0,0x04,0xf8,0xf3]
	%tmp2 = call <4 x i32> @llvm.arm.neon.vclz.v4i32(<4 x i32> %tmp1)
	ret <4 x i32> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vclz.v8i8(<8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vclz.v4i16(<4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vclz.v2i32(<2 x i32>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vclz.v16i8(<16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vclz.v8i16(<8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vclz.v4i32(<4 x i32>) nounwind readnone

define <8 x i8> @vclss8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vcls.s8	d16, d16                @ encoding: [0x20,0x04,0xf0,0xf3]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vcls.v8i8(<8 x i8> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vclss16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vcls.s16	d16, d16        @ encoding: [0x20,0x04,0xf4,0xf3]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vcls.v4i16(<4 x i16> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vclss32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vcls.s32	d16, d16        @ encoding: [0x20,0x04,0xf8,0xf3]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vcls.v2i32(<2 x i32> %tmp1)
	ret <2 x i32> %tmp2
}

define <16 x i8> @vclsQs8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vcls.s8	q8, q8                  @ encoding: [0x60,0x04,0xf0,0xf3]
	%tmp2 = call <16 x i8> @llvm.arm.neon.vcls.v16i8(<16 x i8> %tmp1)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vclsQs16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vcls.s16	q8, q8          @ encoding: [0x60,0x04,0xf4,0xf3]
	%tmp2 = call <8 x i16> @llvm.arm.neon.vcls.v8i16(<8 x i16> %tmp1)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vclsQs32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vcls.s32	q8, q8          @ encoding: [0x60,0x04,0xf8,0xf3]
	%tmp2 = call <4 x i32> @llvm.arm.neon.vcls.v4i32(<4 x i32> %tmp1)
	ret <4 x i32> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vcls.v8i8(<8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vcls.v4i16(<4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vcls.v2i32(<2 x i32>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vcls.v16i8(<16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vcls.v8i16(<8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vcls.v4i32(<4 x i32>) nounwind readnone
