; RUN: opt %s -globalopt | llvm-dis | grep {define void @a}

define internal void @f() {
	ret void
}

@a = alias void ()* @f

define void @g() {
	call void()* @a()
	ret void
}
