; RUN: llc < %s -O0 -fast-isel-abort -march=x86 | FileCheck %s

%struct.s = type {i32, i32, i32}

define i32 @test1() nounwind {
tak:
	%tmp = call i1 @foo()
	br i1 %tmp, label %BB1, label %BB2
BB1:
	ret i32 1
BB2:
	ret i32 0
; CHECK: test1:
; CHECK: calll
; CHECK-NEXT: testb	$1
}
declare i1 @foo() zeroext nounwind

declare void @foo2(%struct.s* byval)

define void @test2(%struct.s* %d) nounwind {
  call void @foo2(%struct.s* %d byval)
  ret void
; CHECK: test2:
; CHECK: movl	(%eax)
; CHECK: movl {{.*}}, (%esp)
; CHECK: movl	4(%eax)
; CHECK: movl {{.*}}, 4(%esp)
; CHECK: movl	8(%eax)
; CHECK: movl {{.*}}, 8(%esp)
}
