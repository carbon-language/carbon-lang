; RUN: llvm-as < %s | llc | grep ly     | count 2
; RUN: llvm-as < %s | llc | grep sty    | count 2
; RUN: llvm-as < %s | llc | grep {l	%}  | count 2
; RUN: llvm-as < %s | llc | grep {st	%} | count 2

target datalayout = "E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-a0:16:16"
target triple = "s390x-linux"

define void @foo1(i32* nocapture %foo, i32* nocapture %bar) nounwind {
entry:
	%tmp1 = load i32* %foo		; <i32> [#uses=1]
	store i32 %tmp1, i32* %bar
	ret void
}

define void @foo2(i32* nocapture %foo, i32* nocapture %bar, i64 %idx) nounwind {
entry:
	%add.ptr = getelementptr i32* %foo, i64 1		; <i32*> [#uses=1]
	%tmp1 = load i32* %add.ptr		; <i32> [#uses=1]
	%add.ptr3.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr5 = getelementptr i32* %bar, i64 %add.ptr3.sum		; <i32*> [#uses=1]
	store i32 %tmp1, i32* %add.ptr5
	ret void
}

define void @foo3(i32* nocapture %foo, i32* nocapture %bar, i64 %idx) nounwind {
entry:
	%sub.ptr = getelementptr i32* %foo, i64 -1		; <i32*> [#uses=1]
	%tmp1 = load i32* %sub.ptr		; <i32> [#uses=1]
	%sub.ptr3.sum = add i64 %idx, -1		; <i64> [#uses=1]
	%add.ptr = getelementptr i32* %bar, i64 %sub.ptr3.sum		; <i32*> [#uses=1]
	store i32 %tmp1, i32* %add.ptr
	ret void
}

define void @foo4(i32* nocapture %foo, i32* nocapture %bar, i64 %idx) nounwind {
entry:
	%add.ptr = getelementptr i32* %foo, i64 8192		; <i32*> [#uses=1]
	%tmp1 = load i32* %add.ptr		; <i32> [#uses=1]
	%add.ptr3.sum = add i64 %idx, 8192		; <i64> [#uses=1]
	%add.ptr5 = getelementptr i32* %bar, i64 %add.ptr3.sum		; <i32*> [#uses=1]
	store i32 %tmp1, i32* %add.ptr5
	ret void
}
