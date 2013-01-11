; RUN: opt < %s -tailcallelim -S | FileCheck %s


declare void @foo()

define void @bar() {
; CHECK: tail call void @foo()
	call void @foo()
	ret void
}

