; RUN: opt < %s -tailcallelim -S | FileCheck %s

declare void @noarg()
declare void @use(i32*)

; Trivial case. Mark @noarg with tail call.
define void @test0() {
; CHECK: tail call void @noarg()
	call void @noarg()
	ret void
}

; PR615. Make sure that we do not move the alloca so that it interferes with the tail call.
define i32 @test1() {
; CHECK: i32 @test1()
; CHECK-NEXT: alloca
	%A = alloca i32		; <i32*> [#uses=2]
	store i32 5, i32* %A
	call void @use(i32* %A)
	%X = tail call i32 @test1()		; <i32> [#uses=1]
	ret i32 %X
}

; This function contains intervening instructions which should be moved out of the way
define i32 @test2(i32 %X) {
; CHECK: i32 @test2
; CHECK-NOT: call
; CHECK: ret i32
entry:
	%tmp.1 = icmp eq i32 %X, 0		; <i1> [#uses=1]
	br i1 %tmp.1, label %then.0, label %endif.0
then.0:		; preds = %entry
	%tmp.4 = add i32 %X, 1		; <i32> [#uses=1]
	ret i32 %tmp.4
endif.0:		; preds = %entry
	%tmp.10 = add i32 %X, -1		; <i32> [#uses=1]
	%tmp.8 = call i32 @test2(i32 %tmp.10)		; <i32> [#uses=1]
	%DUMMY = add i32 %X, 1		; <i32> [#uses=0]
	ret i32 %tmp.8
}

; Though this case seems to be fairly unlikely to occur in the wild, someone
; plunked it into the demo script, so maybe they care about it.
define i32 @test3(i32 %c) {
; CHECK: i32 @test3
; CHECK-NOT: call
; CHECK: ret i32 0
entry:
	%tmp.1 = icmp eq i32 %c, 0		; <i1> [#uses=1]
	br i1 %tmp.1, label %return, label %else
else:		; preds = %entry
	%tmp.5 = add i32 %c, -1		; <i32> [#uses=1]
	%tmp.3 = call i32 @test3(i32 %tmp.5)		; <i32> [#uses=0]
	ret i32 0
return:		; preds = %entry
	ret i32 0
}

