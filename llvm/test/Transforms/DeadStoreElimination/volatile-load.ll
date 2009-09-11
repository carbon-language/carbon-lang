; RUN: opt < %s -dse -S | grep {volatile load}

@g_1 = global i32 0

define void @foo() nounwind  {
	%t = volatile load i32* @g_1
	ret void
}
