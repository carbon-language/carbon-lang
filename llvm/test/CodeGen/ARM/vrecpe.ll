; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <2 x i32> @vrecpei32(<2 x i32>* %A) nounwind {
;CHECK: vrecpei32:
;CHECK: vrecpe.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vrecpe.v2i32(<2 x i32> %tmp1)
	ret <2 x i32> %tmp2
}

define <4 x i32> @vrecpeQi32(<4 x i32>* %A) nounwind {
;CHECK: vrecpeQi32:
;CHECK: vrecpe.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vrecpe.v4i32(<4 x i32> %tmp1)
	ret <4 x i32> %tmp2
}

define <2 x float> @vrecpef32(<2 x float>* %A) nounwind {
;CHECK: vrecpef32:
;CHECK: vrecpe.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = call <2 x float> @llvm.arm.neon.vrecpe.v2f32(<2 x float> %tmp1)
	ret <2 x float> %tmp2
}

define <4 x float> @vrecpeQf32(<4 x float>* %A) nounwind {
;CHECK: vrecpeQf32:
;CHECK: vrecpe.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = call <4 x float> @llvm.arm.neon.vrecpe.v4f32(<4 x float> %tmp1)
	ret <4 x float> %tmp2
}

declare <2 x i32> @llvm.arm.neon.vrecpe.v2i32(<2 x i32>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vrecpe.v4i32(<4 x i32>) nounwind readnone

declare <2 x float> @llvm.arm.neon.vrecpe.v2f32(<2 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vrecpe.v4f32(<4 x float>) nounwind readnone
