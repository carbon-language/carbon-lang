; RUN: opt < %s -jump-threading -loop-rotate -instcombine -indvars -loop-unroll -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S -verify-dom-info -verify-loop-info > %t
; RUN: grep "store volatile" %t | count 3
; RUN: not grep "br label" %t

; Jump threading should not prevent this loop from being unrolled.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"
@v1 = external global i32		; <i32*> [#uses=2]

define i32 @unroll() nounwind {
entry:
	br label %bb4

bb:		; preds = %bb4
	%0 = icmp eq i32 %i.0, 0		; <i1> [#uses=1]
	br i1 %0, label %bb1, label %bb2

bb1:		; preds = %bb
	store volatile i32 1000, i32* @v1, align 4
	br label %bb3

bb2:		; preds = %bb
	store volatile i32 1001, i32* @v1, align 4
	br label %bb3

bb3:		; preds = %bb2, %bb1
	%1 = add i32 %i.0, 1		; <i32> [#uses=1]
	br label %bb4

bb4:		; preds = %bb3, %entry
	%i.0 = phi i32 [ 0, %entry ], [ %1, %bb3 ]		; <i32> [#uses=3]
	%2 = icmp sgt i32 %i.0, 2		; <i1> [#uses=1]
	br i1 %2, label %bb5, label %bb

bb5:		; preds = %bb4
	ret i32 0
}
