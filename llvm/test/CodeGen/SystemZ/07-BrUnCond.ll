; RUN: llc < %s

target datalayout = "E-p:64:64:64-i1:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128"
target triple = "s390x-linux"

define void @foo() noreturn nounwind {
entry:
	tail call void @baz() nounwind
	br label %l1

l1:		; preds = %entry, %l1
	tail call void @bar() nounwind
	br label %l1
}

declare void @bar()

declare void @baz()
