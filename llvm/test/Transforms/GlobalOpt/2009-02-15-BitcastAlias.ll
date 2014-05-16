; RUN: opt < %s -globalopt

@g = global i32 0

@a = alias i8, i32* @g

define void @f() {
	%tmp = load i8* @a
	ret void
}
