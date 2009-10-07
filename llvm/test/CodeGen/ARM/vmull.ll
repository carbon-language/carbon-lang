; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i16> @vmulls8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vmulls8:
;CHECK: vmull.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vmulls.v8i16(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vmulls16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vmulls16:
;CHECK: vmull.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vmulls32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vmulls32:
;CHECK: vmull.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i64> %tmp3
}

define <8 x i16> @vmullu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vmullu8:
;CHECK: vmull.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vmullu.v8i16(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vmullu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vmullu16:
;CHECK: vmull.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vmullu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vmullu32:
;CHECK: vmull.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i64> %tmp3
}

define <8 x i16> @vmullp8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vmullp8:
;CHECK: vmull.p8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vmullp.v8i16(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

declare <8 x i16> @llvm.arm.neon.vmulls.v8i16(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vmullu.v8i16(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i16>  @llvm.arm.neon.vmullp.v8i16(<8 x i8>, <8 x i8>) nounwind readnone
