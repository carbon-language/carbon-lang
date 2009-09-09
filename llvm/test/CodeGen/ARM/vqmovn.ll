; RUN: llc < %s -march=arm -mattr=+neon > %t
; RUN: grep {vqmovn\\.s16} %t | count 1
; RUN: grep {vqmovn\\.s32} %t | count 1
; RUN: grep {vqmovn\\.s64} %t | count 1
; RUN: grep {vqmovn\\.u16} %t | count 1
; RUN: grep {vqmovn\\.u32} %t | count 1
; RUN: grep {vqmovn\\.u64} %t | count 1
; RUN: grep {vqmovun\\.s16} %t | count 1
; RUN: grep {vqmovun\\.s32} %t | count 1
; RUN: grep {vqmovun\\.s64} %t | count 1

define <8 x i8> @vqmovns16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqmovns.v8i8(<8 x i16> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqmovns32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqmovns.v4i16(<4 x i32> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqmovns64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqmovns.v2i32(<2 x i64> %tmp1)
	ret <2 x i32> %tmp2
}

define <8 x i8> @vqmovnu16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqmovnu.v8i8(<8 x i16> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqmovnu32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqmovnu.v4i16(<4 x i32> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqmovnu64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqmovnu.v2i32(<2 x i64> %tmp1)
	ret <2 x i32> %tmp2
}

define <8 x i8> @vqmovuns16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqmovnsu.v8i8(<8 x i16> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqmovuns32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqmovnsu.v4i16(<4 x i32> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqmovuns64(<2 x i64>* %A) nounwind {
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
