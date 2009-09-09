; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <2 x i32> @vacgtf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vacgtf32:
;CHECK: vacgt.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vacgtd(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x i32> %tmp3
}

define <4 x i32> @vacgtQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK: vacgtQf32:
;CHECK: vacgt.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vacgtq(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x i32> %tmp3
}

declare <2 x i32> @llvm.arm.neon.vacgtd(<2 x float>, <2 x float>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vacgtq(<4 x float>, <4 x float>) nounwind readnone
