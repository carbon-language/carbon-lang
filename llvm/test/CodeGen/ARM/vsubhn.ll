; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vsubhni16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vsubhni16:
;CHECK: vsubhn.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vsubhn.v8i8(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vsubhni32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vsubhni32:
;CHECK: vsubhn.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vsubhn.v4i16(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vsubhni64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK: vsubhni64:
;CHECK: vsubhn.i64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vsubhn.v2i32(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vsubhn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vsubhn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vsubhn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone
