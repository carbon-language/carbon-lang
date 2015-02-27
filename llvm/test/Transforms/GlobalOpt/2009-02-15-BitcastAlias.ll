; RUN: opt < %s -globalopt

@g = global i32 0

@a = alias bitcast (i32* @g to i8*)

define void @f() {
	%tmp = load i8, i8* @a
	ret void
}
