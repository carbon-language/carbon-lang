; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vclz8(<8 x i8>* %A) nounwind {
;CHECK: vclz8:
;CHECK: vclz.i8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vclz.v8i8(<8 x i8> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vclz16(<4 x i16>* %A) nounwind {
;CHECK: vclz16:
;CHECK: vclz.i16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vclz.v4i16(<4 x i16> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vclz32(<2 x i32>* %A) nounwind {
;CHECK: vclz32:
;CHECK: vclz.i32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vclz.v2i32(<2 x i32> %tmp1)
	ret <2 x i32> %tmp2
}

define <16 x i8> @vclzQ8(<16 x i8>* %A) nounwind {
;CHECK: vclzQ8:
;CHECK: vclz.i8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = call <16 x i8> @llvm.arm.neon.vclz.v16i8(<16 x i8> %tmp1)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vclzQ16(<8 x i16>* %A) nounwind {
;CHECK: vclzQ16:
;CHECK: vclz.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vclz.v8i16(<8 x i16> %tmp1)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vclzQ32(<4 x i32>* %A) nounwind {
;CHECK: vclzQ32:
;CHECK: vclz.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vclz.v4i32(<4 x i32> %tmp1)
	ret <4 x i32> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vclz.v8i8(<8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vclz.v4i16(<4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vclz.v2i32(<2 x i32>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vclz.v16i8(<16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vclz.v8i16(<8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vclz.v4i32(<4 x i32>) nounwind readnone
