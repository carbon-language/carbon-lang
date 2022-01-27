; RUN: llc < %s -mtriple=i686-- | FileCheck %s

define void @foo(float* nocapture %A, float* nocapture %B, float* nocapture %C, i32 %N) nounwind {
; CHECK: foo
; CHECK: incl

entry:
	%0 = icmp sgt i32 %N, 0		; <i1> [#uses=1]
	br i1 %0, label %bb, label %return

bb:		; preds = %bb, %entry
	%i.03 = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]		; <i32> [#uses=5]
	%1 = getelementptr float, float* %A, i32 %i.03		; <float*> [#uses=1]
	%2 = load float, float* %1, align 4		; <float> [#uses=1]
	%3 = getelementptr float, float* %B, i32 %i.03		; <float*> [#uses=1]
	%4 = load float, float* %3, align 4		; <float> [#uses=1]
	%5 = fadd float %2, %4		; <float> [#uses=1]
	%6 = getelementptr float, float* %C, i32 %i.03		; <float*> [#uses=1]
	store float %5, float* %6, align 4
	%7 = add i32 %i.03, 10		; <i32> [#uses=3]
	%8 = getelementptr float, float* %A, i32 %7		; <float*> [#uses=1]
	%9 = load float, float* %8, align 4		; <float> [#uses=1]
	%10 = getelementptr float, float* %B, i32 %7		; <float*> [#uses=1]
	%11 = load float, float* %10, align 4		; <float> [#uses=1]
	%12 = fadd float %9, %11		; <float> [#uses=1]
	%13 = getelementptr float, float* %C, i32 %7		; <float*> [#uses=1]
	store float %12, float* %13, align 4
	%indvar.next = add i32 %i.03, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %N		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb, %entry
	ret void
}
