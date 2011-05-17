; RUN: opt -instcombine -S < %s | FileCheck %s
; <rdar://problem/6945110>

define <4 x i32> @kernel3_vertical(<4 x i16> * %src, <8 x i16> * %foo) nounwind {
entry:
	%tmp = load <4 x i16>* %src
	%tmp1 = load <8 x i16>* %foo
; CHECK: shufflevector
	%tmp2 = shufflevector <4 x i16> %tmp, <4 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK-NOT: shufflevector
	%tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 4, i32 5, i32 6, i32 7>
	%0 = call <4 x i32> @llvm.x86.sse41.pmovzxwd(<8 x i16> %tmp3)
	ret <4 x i32> %0
}
declare <4 x i32> @llvm.x86.sse41.pmovzxwd(<8 x i16>) nounwind readnone
