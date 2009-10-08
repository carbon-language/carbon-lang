; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vqmovns16(<8 x i16>* %A) nounwind {
;CHECK: vqmovns16:
;CHECK: vqmovn.s16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqmovns.v8i8(<8 x i16> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqmovns32(<4 x i32>* %A) nounwind {
;CHECK: vqmovns32:
;CHECK: vqmovn.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqmovns.v4i16(<4 x i32> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqmovns64(<2 x i64>* %A) nounwind {
;CHECK: vqmovns64:
;CHECK: vqmovn.s64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqmovns.v2i32(<2 x i64> %tmp1)
	ret <2 x i32> %tmp2
}

define <8 x i8> @vqmovnu16(<8 x i16>* %A) nounwind {
;CHECK: vqmovnu16:
;CHECK: vqmovn.u16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqmovnu.v8i8(<8 x i16> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqmovnu32(<4 x i32>* %A) nounwind {
;CHECK: vqmovnu32:
;CHECK: vqmovn.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqmovnu.v4i16(<4 x i32> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqmovnu64(<2 x i64>* %A) nounwind {
;CHECK: vqmovnu64:
;CHECK: vqmovn.u64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqmovnu.v2i32(<2 x i64> %tmp1)
	ret <2 x i32> %tmp2
}

define <8 x i8> @vqmovuns16(<8 x i16>* %A) nounwind {
;CHECK: vqmovuns16:
;CHECK: vqmovun.s16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqmovnsu.v8i8(<8 x i16> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqmovuns32(<4 x i32>* %A) nounwind {
;CHECK: vqmovuns32:
;CHECK: vqmovun.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqmovnsu.v4i16(<4 x i32> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqmovuns64(<2 x i64>* %A) nounwind {
;CHECK: vqmovuns64:
;CHECK: vqmovun.s64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqmovnsu.v2i32(<2 x i64> %tmp1)
	ret <2 x i32> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vqmovns.v8i8(<8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqmovns.v4i16(<4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqmovns.v2i32(<2 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vqmovnu.v8i8(<8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqmovnu.v4i16(<4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqmovnu.v2i32(<2 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vqmovnsu.v8i8(<8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqmovnsu.v4i16(<4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqmovnsu.v2i32(<2 x i64>) nounwind readnone
