; RUN: llc < %s -relocation-model=pic | grep GOTENT | count 6

target datalayout = "E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-a0:16:16"
target triple = "s390x-linux"
@src = external global i32		; <i32*> [#uses=2]
@dst = external global i32		; <i32*> [#uses=2]
@ptr = external global i32*		; <i32**> [#uses=2]

define void @foo1() nounwind {
entry:
	%tmp = load i32* @src		; <i32> [#uses=1]
	store i32 %tmp, i32* @dst
	ret void
}

define void @foo2() nounwind {
entry:
	store i32* @dst, i32** @ptr
	ret void
}

define void @foo3() nounwind {
entry:
	%tmp = load i32* @src		; <i32> [#uses=1]
	%tmp1 = load i32** @ptr		; <i32*> [#uses=1]
	%arrayidx = getelementptr i32* %tmp1, i64 1		; <i32*> [#uses=1]
	store i32 %tmp, i32* %arrayidx
	ret void
}
