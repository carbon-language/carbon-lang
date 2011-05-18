; RUN: llc < %s -march=x86-64 -mattr=sse41 | FileCheck %s
; RUN: opt -std-compile-opts < %s | llc -march=x86-64 -mattr=sse41 | FileCheck --check-prefix=CHECK_OPT_LLC %s

define <8 x i16> @shuf6(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
; CHECK: pshufb
; CHECK-NOT: pshufb
; CHECK: ret
entry:
  %tmp9 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 3, i32 2, i32 0, i32 2, i32 1, i32 5, i32 6 , i32 undef >
  ret <8 x i16> %tmp9
}

define <8 x i16> @shuf7(<8 x i16> %t0) {
; CHECK: pshufd
  %tmp10 = shufflevector <8 x i16> %t0, <8 x i16> undef, <8 x i32> < i32 undef, i32 2, i32 2, i32 2, i32 2, i32 2, i32 undef, i32 undef >
  ret <8 x i16> %tmp10
}


; <rdar://problem/6945110>
define <4 x i32> @kernel3_vertical(<4 x i16> * %src, <8 x i16> * %foo) nounwind {
entry:
; CHECK_OPT_LLC: call{{.*nothing}}
        call void @nothing()
	%tmp = load <4 x i16>* %src
	%tmp1 = load <8 x i16>* %foo
; pmovzxwd ignores the upper 64-bits of its input; everything between the call and pmovzxwd should be removed.
	%tmp2 = shufflevector <4 x i16> %tmp, <4 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
	%tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 4, i32 5, i32 6, i32 7>
; CHECK_OPT_LLC-NEXT: pmovzxwd
	%0 = call <4 x i32> @llvm.x86.sse41.pmovzxwd(<8 x i16> %tmp3)
	ret <4 x i32> %0
}
declare void @nothing() nounwind
declare <4 x i32> @llvm.x86.sse41.pmovzxwd(<8 x i16>) nounwind readnone
