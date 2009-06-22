; RUN: llvm-as < %s | llc -march=arm -mattr=+neon > %t
; RUN: grep {vcvt\\.s32\\.f32} %t | count 2
; RUN: grep {vcvt\\.u32\\.f32} %t | count 2
; RUN: grep {vcvt\\.f32\\.s32} %t | count 2
; RUN: grep {vcvt\\.f32\\.u32} %t | count 2

define <2 x i32> @vcvt_f32tos32(<2 x float>* %A) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vcvtfp2fxs.v2i32.v2f32(<2 x float> %tmp1, i32 1)
	ret <2 x i32> %tmp2
}

define <2 x i32> @vcvt_f32tou32(<2 x float>* %A) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vcvtfp2fxu.v2i32.v2f32(<2 x float> %tmp1, i32 1)
	ret <2 x i32> %tmp2
}

define <2 x float> @vcvt_s32tof32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <2 x float> @llvm.arm.neon.vcvtfxs2fp.v2f32.v2i32(<2 x i32> %tmp1, i32 1)
	ret <2 x float> %tmp2
}

define <2 x float> @vcvt_u32tof32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <2 x float> @llvm.arm.neon.vcvtfxu2fp.v2f32.v2i32(<2 x i32> %tmp1, i32 1)
	ret <2 x float> %tmp2
}

declare <2 x i32> @llvm.arm.neon.vcvtfp2fxs.v2i32.v2f32(<2 x float>, i32) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vcvtfp2fxu.v2i32.v2f32(<2 x float>, i32) nounwind readnone
declare <2 x float> @llvm.arm.neon.vcvtfxs2fp.v2f32.v2i32(<2 x i32>, i32) nounwind readnone
declare <2 x float> @llvm.arm.neon.vcvtfxu2fp.v2f32.v2i32(<2 x i32>, i32) nounwind readnone

define <4 x i32> @vcvtQ_f32tos32(<4 x float>* %A) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vcvtfp2fxs.v4i32.v4f32(<4 x float> %tmp1, i32 1)
	ret <4 x i32> %tmp2
}

define <4 x i32> @vcvtQ_f32tou32(<4 x float>* %A) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vcvtfp2fxu.v4i32.v4f32(<4 x float> %tmp1, i32 1)
	ret <4 x i32> %tmp2
}

define <4 x float> @vcvtQ_s32tof32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x float> @llvm.arm.neon.vcvtfxs2fp.v4f32.v4i32(<4 x i32> %tmp1, i32 1)
	ret <4 x float> %tmp2
}

define <4 x float> @vcvtQ_u32tof32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x float> @llvm.arm.neon.vcvtfxu2fp.v4f32.v4i32(<4 x i32> %tmp1, i32 1)
	ret <4 x float> %tmp2
}

declare <4 x i32> @llvm.arm.neon.vcvtfp2fxs.v4i32.v4f32(<4 x float>, i32) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vcvtfp2fxu.v4i32.v4f32(<4 x float>, i32) nounwind readnone
declare <4 x float> @llvm.arm.neon.vcvtfxs2fp.v4f32.v4i32(<4 x i32>, i32) nounwind readnone
declare <4 x float> @llvm.arm.neon.vcvtfxu2fp.v4f32.v4i32(<4 x i32>, i32) nounwind readnone

