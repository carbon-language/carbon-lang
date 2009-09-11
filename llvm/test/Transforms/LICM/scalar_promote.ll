; RUN: opt < %s  -licm -S | FileCheck %s
@X = global i32 7		; <i32*> [#uses=4]

define void @test1(i32 %i) {
Entry:
	br label %Loop
; CHECK: @test1
; CHECK: Entry:
; CHECK-NEXT:   load i32* @X
; CHECK-NEXT:   br label %Loop


Loop:		; preds = %Loop, %0
	%j = phi i32 [ 0, %Entry ], [ %Next, %Loop ]		; <i32> [#uses=1]
	%x = load i32* @X		; <i32> [#uses=1]
	%x2 = add i32 %x, 1		; <i32> [#uses=1]
	store i32 %x2, i32* @X
	%Next = add i32 %j, 1		; <i32> [#uses=2]
	%cond = icmp eq i32 %Next, 0		; <i1> [#uses=1]
	br i1 %cond, label %Out, label %Loop

Out:	
	ret void
; CHECK: Out:
; CHECK-NEXT:   store i32 %x2, i32* @X
; CHECK-NEXT:   ret void

}

define void @test2(i32 %i) {
Entry:
	br label %Loop
; CHECK: @test2
; CHECK: Entry:
; CHECK-NEXT:  %X1 = getelementptr i32* @X, i64 0 
; CHECK-NEXT:    %X2 = getelementptr i32* @X, i64 0
; CHECK-NEXT:    %X1.promoted = load i32* %X1 
; CHECK-NEXT:    br label %Loop

Loop:		; preds = %Loop, %0
	%X1 = getelementptr i32* @X, i64 0		; <i32*> [#uses=1]
	%A = load i32* %X1		; <i32> [#uses=1]
	%V = add i32 %A, 1		; <i32> [#uses=1]
	%X2 = getelementptr i32* @X, i64 0		; <i32*> [#uses=1]
	store i32 %V, i32* %X2
	br i1 false, label %Loop, label %Exit

Exit:		; preds = %Loop
	ret void
; CHECK: Exit:
; CHECK-NEXT:   store i32 %V, i32* %X1
; CHECK-NEXT:   ret void
}



define void @test3(i32 %i) {
; CHECK: @test3
	br label %Loop
Loop:
        ; Should not promote this to a register
	%x = volatile load i32* @X
	%x2 = add i32 %x, 1	
	store i32 %x2, i32* @X
	br i1 true, label %Out, label %Loop
        
; CHECK: Loop:
; CHECK-NEXT: volatile load

Out:		; preds = %Loop
	ret void
}

