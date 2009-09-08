; RUN: opt %s -prune-eh | llvm-dis | not grep nounwind

define weak void @f() {
entry:
        ret void
}

define void @g() {
entry:
	call void @f()
	ret void
}
