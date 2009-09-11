; RUN: opt < %s -licm -S | FileCheck %s

@X = global i32 0		; <i32*> [#uses=1]

declare void @foo()

; This testcase tests for a problem where LICM hoists 
; potentially trapping instructions when they are not guaranteed to execute.
define i32 @test1(i1 %c) {
; CHECK: @test1
	%A = load i32* @X		; <i32> [#uses=2]
	br label %Loop
Loop:		; preds = %LoopTail, %0
	call void @foo( )
	br i1 %c, label %LoopTail, label %IfUnEqual
        
IfUnEqual:		; preds = %Loop
; CHECK: IfUnEqual:
; CHECK-NEXT: sdiv i32 4, %A
	%B1 = sdiv i32 4, %A		; <i32> [#uses=1]
	br label %LoopTail
        
LoopTail:		; preds = %IfUnEqual, %Loop
	%B = phi i32 [ 0, %Loop ], [ %B1, %IfUnEqual ]		; <i32> [#uses=1]
	br i1 %c, label %Loop, label %Out
Out:		; preds = %LoopTail
	%C = sub i32 %A, %B		; <i32> [#uses=1]
	ret i32 %C
}


declare void @foo2(i32)


;; It is ok and desirable to hoist this potentially trapping instruction.
define i32 @test2(i1 %c) {
; CHECK: @test2
; CHECK-NEXT: load i32* @X
; CHECK-NEXT: %B = sdiv i32 4, %A
	%A = load i32* @X		; <i32> [#uses=2]
	br label %Loop
Loop:
        ;; Should have hoisted this div!
	%B = sdiv i32 4, %A		; <i32> [#uses=2]
	call void @foo2( i32 %B )
	br i1 %c, label %Loop, label %Out
Out:		; preds = %Loop
	%C = sub i32 %A, %B		; <i32> [#uses=1]
	ret i32 %C
}
