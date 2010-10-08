; RUN: llc < %s -march=x86-64 | FileCheck %s

define i32 @test1(i32 %X, i32* %y) nounwind {
	%tmp = load i32* %y		; <i32> [#uses=1]
	%tmp.upgrd.1 = icmp eq i32 %tmp, 0		; <i1> [#uses=1]
	br i1 %tmp.upgrd.1, label %ReturnBlock, label %cond_true

cond_true:		; preds = %0
	ret i32 1

ReturnBlock:		; preds = %0
	ret i32 0
; CHECK: test1:
; CHECK: cmpl	$0, (%rsi)
}

define i32 @test2(i32 %X, i32* %y) nounwind {
	%tmp = load i32* %y		; <i32> [#uses=1]
	%tmp1 = shl i32 %tmp, 3		; <i32> [#uses=1]
	%tmp1.upgrd.2 = icmp eq i32 %tmp1, 0		; <i1> [#uses=1]
	br i1 %tmp1.upgrd.2, label %ReturnBlock, label %cond_true

cond_true:		; preds = %0
	ret i32 1

ReturnBlock:		; preds = %0
	ret i32 0
; CHECK: test2:
; CHECK: movl	(%rsi), %eax
; CHECK: shll	$3, %eax
; CHECK: testl	%eax, %eax
}
