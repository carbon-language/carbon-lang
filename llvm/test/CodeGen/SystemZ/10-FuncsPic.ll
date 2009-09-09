; RUN: llc < %s -relocation-model=pic | grep GOTENT | count 3
; RUN: llc < %s -relocation-model=pic | grep PLT | count 1

target datalayout = "E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-a0:16:16"
target triple = "s390x-linux"
@ptr = external global void (...)*		; <void (...)**> [#uses=2]

define void @foo1() nounwind {
entry:
	store void (...)* @func, void (...)** @ptr
	ret void
}

declare void @func(...)

define void @foo2() nounwind {
entry:
	tail call void (...)* @func() nounwind
	ret void
}

define void @foo3() nounwind {
entry:
	%tmp = load void (...)** @ptr		; <void (...)*> [#uses=1]
	tail call void (...)* %tmp() nounwind
	ret void
}
