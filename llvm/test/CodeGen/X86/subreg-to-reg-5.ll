; RUN: llc < %s -march=x86-64 > %t
; RUN: grep addl %t
; RUN: not egrep {movl|movq} %t

define float @foo(float* %B) nounwind {
entry:
	br label %bb2

bb2:		; preds = %bb3, %entry
	%B_addr.0.rec = phi i64 [ %indvar.next154, %bb3 ], [ 0, %entry ]		; <i64> [#uses=2]
        %z = icmp slt i64 %B_addr.0.rec, 20000
	br i1 %z, label %bb3, label %bb4

bb3:		; preds = %bb2
	%indvar.next154 = add i64 %B_addr.0.rec, 1		; <i64> [#uses=1]
	br label %bb2

bb4:		; preds = %bb2
	%B_addr.0 = getelementptr float* %B, i64 %B_addr.0.rec		; <float*> [#uses=1]
	%t1 = ptrtoint float* %B_addr.0 to i64		; <i64> [#uses=1]
	%t2 = and i64 %t1, 4294967295		; <i64> [#uses=1]
	%t3 = icmp eq i64 %t2, 0		; <i1> [#uses=1]
	br i1 %t3, label %bb5, label %bb10.preheader

bb10.preheader:		; preds = %bb4
	br label %bb9

bb5:		; preds = %bb4
	ret float 7.0

bb9:		; preds = %bb10.preheader
	%t5 = getelementptr float* %B, i64 0		; <float*> [#uses=1]
	%t7 = load float* %t5		; <float> [#uses=1]
	ret float %t7
}
