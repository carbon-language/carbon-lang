; RUN: llc < %s | grep {sthy.%} | count 2
; RUN: llc < %s | grep {lhy.%}  | count 2
; RUN: llc < %s | grep {lh.%}   | count 6
; RUN: llc < %s | grep {sth.%}  | count 2

target datalayout = "E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-a0:16:16"
target triple = "s390x-linux"

define void @foo1(i16* nocapture %foo, i16* nocapture %bar) nounwind {
entry:
	%tmp1 = load i16* %foo		; <i16> [#uses=1]
	store i16 %tmp1, i16* %bar
	ret void
}

define void @foo2(i16* nocapture %foo, i16* nocapture %bar, i64 %idx) nounwind {
entry:
	%add.ptr = getelementptr i16* %foo, i64 1		; <i16*> [#uses=1]
	%tmp1 = load i16* %add.ptr		; <i16> [#uses=1]
	%add.ptr3.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr5 = getelementptr i16* %bar, i64 %add.ptr3.sum		; <i16*> [#uses=1]
	store i16 %tmp1, i16* %add.ptr5
	ret void
}

define void @foo3(i16* nocapture %foo, i16* nocapture %bar, i64 %idx) nounwind {
entry:
	%sub.ptr = getelementptr i16* %foo, i64 -1		; <i16*> [#uses=1]
	%tmp1 = load i16* %sub.ptr		; <i16> [#uses=1]
	%sub.ptr3.sum = add i64 %idx, -1		; <i64> [#uses=1]
	%add.ptr = getelementptr i16* %bar, i64 %sub.ptr3.sum		; <i16*> [#uses=1]
	store i16 %tmp1, i16* %add.ptr
	ret void
}

define void @foo4(i16* nocapture %foo, i16* nocapture %bar, i64 %idx) nounwind {
entry:
	%add.ptr = getelementptr i16* %foo, i64 8192		; <i16*> [#uses=1]
	%tmp1 = load i16* %add.ptr		; <i16> [#uses=1]
	%add.ptr3.sum = add i64 %idx, 8192		; <i64> [#uses=1]
	%add.ptr5 = getelementptr i16* %bar, i64 %add.ptr3.sum		; <i16*> [#uses=1]
	store i16 %tmp1, i16* %add.ptr5
	ret void
}

define void @foo5(i16* nocapture %foo, i32* nocapture %bar) nounwind {
entry:
	%tmp1 = load i16* %foo		; <i16> [#uses=1]
	%conv = sext i16 %tmp1 to i32		; <i32> [#uses=1]
	store i32 %conv, i32* %bar
	ret void
}

define void @foo6(i16* nocapture %foo, i32* nocapture %bar, i64 %idx) nounwind {
entry:
	%add.ptr = getelementptr i16* %foo, i64 1		; <i16*> [#uses=1]
	%tmp1 = load i16* %add.ptr		; <i16> [#uses=1]
	%conv = sext i16 %tmp1 to i32		; <i32> [#uses=1]
	%add.ptr3.sum = add i64 %idx, 1		; <i64> [#uses=1]
	%add.ptr5 = getelementptr i32* %bar, i64 %add.ptr3.sum		; <i32*> [#uses=1]
	store i32 %conv, i32* %add.ptr5
	ret void
}

define void @foo7(i16* nocapture %foo, i32* nocapture %bar, i64 %idx) nounwind {
entry:
	%sub.ptr = getelementptr i16* %foo, i64 -1		; <i16*> [#uses=1]
	%tmp1 = load i16* %sub.ptr		; <i16> [#uses=1]
	%conv = sext i16 %tmp1 to i32		; <i32> [#uses=1]
	%sub.ptr3.sum = add i64 %idx, -1		; <i64> [#uses=1]
	%add.ptr = getelementptr i32* %bar, i64 %sub.ptr3.sum		; <i32*> [#uses=1]
	store i32 %conv, i32* %add.ptr
	ret void
}

define void @foo8(i16* nocapture %foo, i32* nocapture %bar, i64 %idx) nounwind {
entry:
	%add.ptr = getelementptr i16* %foo, i64 8192		; <i16*> [#uses=1]
	%tmp1 = load i16* %add.ptr		; <i16> [#uses=1]
	%conv = sext i16 %tmp1 to i32		; <i32> [#uses=1]
	%add.ptr3.sum = add i64 %idx, 8192		; <i64> [#uses=1]
	%add.ptr5 = getelementptr i32* %bar, i64 %add.ptr3.sum		; <i32*> [#uses=1]
	store i32 %conv, i32* %add.ptr5
	ret void
}
