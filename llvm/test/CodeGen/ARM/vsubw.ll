; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i16> @vsubws8(<8 x i16>* %A, <8 x i8>* %B) nounwind {
;CHECK: vsubws8:
;CHECK: vsubw.s8
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vsubws.v8i16(<8 x i16> %tmp1, <8 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vsubws16(<4 x i32>* %A, <4 x i16>* %B) nounwind {
;CHECK: vsubws16:
;CHECK: vsubw.s16
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vsubws.v4i32(<4 x i32> %tmp1, <4 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vsubws32(<2 x i64>* %A, <2 x i32>* %B) nounwind {
;CHECK: vsubws32:
;CHECK: vsubw.s32
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vsubws.v2i64(<2 x i64> %tmp1, <2 x i32> %tmp2)
	ret <2 x i64> %tmp3
}

define <8 x i16> @vsubwu8(<8 x i16>* %A, <8 x i8>* %B) nounwind {
;CHECK: vsubwu8:
;CHECK: vsubw.u8
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vsubwu.v8i16(<8 x i16> %tmp1, <8 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vsubwu16(<4 x i32>* %A, <4 x i16>* %B) nounwind {
;CHECK: vsubwu16:
;CHECK: vsubw.u16
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vsubwu.v4i32(<4 x i32> %tmp1, <4 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vsubwu32(<2 x i64>* %A, <2 x i32>* %B) nounwind {
;CHECK: vsubwu32:
;CHECK: vsubw.u32
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vsubwu.v2i64(<2 x i64> %tmp1, <2 x i32> %tmp2)
	ret <2 x i64> %tmp3
}

declare <8 x i16> @llvm.arm.neon.vsubws.v8i16(<8 x i16>, <8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vsubws.v4i32(<4 x i32>, <4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vsubws.v2i64(<2 x i64>, <2 x i32>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vsubwu.v8i16(<8 x i16>, <8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vsubwu.v4i32(<4 x i32>, <4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vsubwu.v2i64(<2 x i64>, <2 x i32>) nounwind readnone
