; RUN: llc < %s -fast-isel -march=x86 | FileCheck %s

define i32 @t() nounwind {
tak:
	%tmp = call i1 @foo()
	br i1 %tmp, label %BB1, label %BB2
BB1:
	ret i32 1
BB2:
	ret i32 0
; CHECK: calll
; CHECK-NEXT: testb	$1
}

declare i1 @foo() zeroext nounwind
