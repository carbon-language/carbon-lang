; RUN: llc < %s | grep je  | count 1
; RUN: llc < %s | grep jne | count 1
; RUN: llc < %s | grep jhe | count 2
; RUN: llc < %s | grep jle | count 2
; RUN: llc < %s | grep jh  | count 4
; RUN: llc < %s | grep jl  | count 4


target datalayout = "E-p:64:64:64-i1:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128"
target triple = "s390x-unknown-linux-gnu"

define void @foo(i32 %a, i32 %b) nounwind {
entry:
	%cmp = icmp ult i32 %a, %b		; <i1> [#uses=1]
	br i1 %cmp, label %if.then, label %if.end

if.then:		; preds = %entry
	tail call void @bar() nounwind
	ret void

if.end:		; preds = %entry
	ret void
}

declare void @bar()

define void @foo1(i32 %a, i32 %b) nounwind {
entry:
	%cmp = icmp ugt i32 %a, %b		; <i1> [#uses=1]
	br i1 %cmp, label %if.then, label %if.end

if.then:		; preds = %entry
	tail call void @bar() nounwind
	ret void

if.end:		; preds = %entry
	ret void
}

define void @foo2(i32 %a, i32 %b) nounwind {
entry:
	%cmp = icmp ugt i32 %a, %b		; <i1> [#uses=1]
	br i1 %cmp, label %if.end, label %if.then

if.then:		; preds = %entry
	tail call void @bar() nounwind
	ret void

if.end:		; preds = %entry
	ret void
}

define void @foo3(i32 %a, i32 %b) nounwind {
entry:
	%cmp = icmp ult i32 %a, %b		; <i1> [#uses=1]
	br i1 %cmp, label %if.end, label %if.then

if.then:		; preds = %entry
	tail call void @bar() nounwind
	ret void

if.end:		; preds = %entry
	ret void
}

define void @foo4(i32 %a, i32 %b) nounwind {
entry:
	%cmp = icmp eq i32 %a, %b		; <i1> [#uses=1]
	br i1 %cmp, label %if.then, label %if.end

if.then:		; preds = %entry
	tail call void @bar() nounwind
	ret void

if.end:		; preds = %entry
	ret void
}

define void @foo5(i32 %a, i32 %b) nounwind {
entry:
	%cmp = icmp eq i32 %a, %b		; <i1> [#uses=1]
	br i1 %cmp, label %if.end, label %if.then

if.then:		; preds = %entry
	tail call void @bar() nounwind
	ret void

if.end:		; preds = %entry
	ret void
}

define void @foo6(i32 %a, i32 %b) nounwind {
entry:
	%cmp = icmp slt i32 %a, %b		; <i1> [#uses=1]
	br i1 %cmp, label %if.then, label %if.end

if.then:		; preds = %entry
	tail call void @bar() nounwind
	ret void

if.end:		; preds = %entry
	ret void
}

define void @foo7(i32 %a, i32 %b) nounwind {
entry:
	%cmp = icmp sgt i32 %a, %b		; <i1> [#uses=1]
	br i1 %cmp, label %if.then, label %if.end

if.then:		; preds = %entry
	tail call void @bar() nounwind
	ret void

if.end:		; preds = %entry
	ret void
}

define void @foo8(i32 %a, i32 %b) nounwind {
entry:
	%cmp = icmp sgt i32 %a, %b		; <i1> [#uses=1]
	br i1 %cmp, label %if.end, label %if.then

if.then:		; preds = %entry
	tail call void @bar() nounwind
	ret void

if.end:		; preds = %entry
	ret void
}

define void @foo9(i32 %a, i32 %b) nounwind {
entry:
	%cmp = icmp slt i32 %a, %b		; <i1> [#uses=1]
	br i1 %cmp, label %if.end, label %if.then

if.then:		; preds = %entry
	tail call void @bar() nounwind
	ret void

if.end:		; preds = %entry
	ret void
}
