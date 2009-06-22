; RUN: llvm-as < %s | llc -march=arm -mattr=+neon > %t
; RUN: grep {vcvt\\.s32\\.f32} %t | count 2
; RUN: grep {vcvt\\.u32\\.f32} %t | count 2
; RUN: grep {vcvt\\.f32\\.s32} %t | count 2
; RUN: grep {vcvt\\.f32\\.u32} %t | count 2

define <2 x i32> @vcvt_f32tos32(<2 x float>* %A) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = fptosi <2 x float> %tmp1 to <2 x i32>
	ret <2 x i32> %tmp2
}

define <2 x i32> @vcvt_f32tou32(<2 x float>* %A) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = fptoui <2 x float> %tmp1 to <2 x i32>
	ret <2 x i32> %tmp2
}

define <2 x float> @vcvt_s32tof32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = sitofp <2 x i32> %tmp1 to <2 x float>
	ret <2 x float> %tmp2
}

define <2 x float> @vcvt_u32tof32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = uitofp <2 x i32> %tmp1 to <2 x float>
	ret <2 x float> %tmp2
}

define <4 x i32> @vcvtQ_f32tos32(<4 x float>* %A) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = fptosi <4 x float> %tmp1 to <4 x i32>
	ret <4 x i32> %tmp2
}

define <4 x i32> @vcvtQ_f32tou32(<4 x float>* %A) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = fptoui <4 x float> %tmp1 to <4 x i32>
	ret <4 x i32> %tmp2
}

define <4 x float> @vcvtQ_s32tof32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = sitofp <4 x i32> %tmp1 to <4 x float>
	ret <4 x float> %tmp2
}

define <4 x float> @vcvtQ_u32tof32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = uitofp <4 x i32> %tmp1 to <4 x float>
	ret <4 x float> %tmp2
}
