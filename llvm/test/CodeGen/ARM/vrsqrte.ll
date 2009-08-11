; RUN: llvm-as < %s | llc -march=arm -mattr=+neon > %t
; RUN: grep {vrsqrte\\.u32} %t | count 2
; RUN: grep {vrsqrte\\.f32} %t | count 2

define <2 x i32> @vrsqrtei32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vrsqrte.v2i32(<2 x i32> %tmp1)
	ret <2 x i32> %tmp2
}

define <4 x i32> @vrsqrteQi32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vrsqrte.v4i32(<4 x i32> %tmp1)
	ret <4 x i32> %tmp2
}

define <2 x float> @vrsqrtef32(<2 x float>* %A) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = call <2 x float> @llvm.arm.neon.vrsqrte.v2f32(<2 x float> %tmp1)
	ret <2 x float> %tmp2
}

define <4 x float> @vrsqrteQf32(<4 x float>* %A) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = call <4 x float> @llvm.arm.neon.vrsqrte.v4f32(<4 x float> %tmp1)
	ret <4 x float> %tmp2
}

declare <2 x i32> @llvm.arm.neon.vrsqrte.v2i32(<2 x i32>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vrsqrte.v4i32(<4 x i32>) nounwind readnone

declare <2 x float> @llvm.arm.neon.vrsqrte.v2f32(<2 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vrsqrte.v4f32(<4 x float>) nounwind readnone
