; RUN: llvm-as < %s | llc -march=x86 -mattr=sse41 | grep insertps | count 2

define <4 x float> @t1(<4 x float> %t1, <4 x float> %t2) nounwind {
        %tmp1 = call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %t1, <4 x float> %t2, i32 1) nounwind readnone
        ret <4 x float> %tmp1
}

declare <4 x float> @llvm.x86.sse41.insertps(<4 x float>, <4 x float>, i32) nounwind readnone

define <4 x float> @t2(<4 x float> %t1, float %t2) nounwind {
        %tmp1 = insertelement <4 x float> %t1, float %t2, i32 0
        ret <4 x float> %tmp1
}