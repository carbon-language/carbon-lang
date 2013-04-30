; RUN: llc < %s -march=x86-64 | FileCheck %s

; CHECK: test3
; CHECK: movd
; CHECK-NOT: movd
; CHECK: {{movlhps.*%xmm0, %xmm0}}
; CHECK-NEXT: ret

define <2 x i64> @test3(i64 %A) nounwind {
entry:
	%B = insertelement <2 x i64> undef, i64 %A, i32 1
	ret <2 x i64> %B
}

