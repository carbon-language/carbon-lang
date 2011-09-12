; RUN: opt < %s -indvars -S -enable-iv-rewrite | FileCheck %s
; CHECK: define void @foo
; CHECK: mul
; CHECK: mul
; CHECK: mul
; CHECK: add
; CHECK: sub
; CHECK: define void @bar
; CHECK: mul
; CHECK: mul
; CHECK: mul
; CHECK: add
; CHECK: sub

define void @foo(i64 %n, i64 %m, i64 %o, double* nocapture %p) nounwind {
entry:
	%tmp = icmp sgt i64 %n, 0		; <i1> [#uses=1]
	br i1 %tmp, label %bb.nph, label %return

bb.nph:		; preds = %entry
	%tmp1 = mul i64 %n, 37		; <i64> [#uses=1]
	%tmp2 = mul i64 %tmp1, %m		; <i64> [#uses=1]
	%tmp3 = mul i64 %tmp2, %o		; <i64> [#uses=1]
	br label %bb

bb:		; preds = %bb, %bb.nph
	%i.01 = phi i64 [ %tmp3, %bb.nph ], [ %tmp13, %bb ]		; <i64> [#uses=3]
	%tmp9 = getelementptr double* %p, i64 %i.01		; <double*> [#uses=1]
	%tmp10 = load double* %tmp9, align 8		; <double> [#uses=1]
	%tmp11 = fdiv double %tmp10, 2.100000e+00		; <double> [#uses=1]
	store double %tmp11, double* %tmp9, align 8
	%tmp13 = add i64 %i.01, 1		; <i64> [#uses=2]
	%tmp14 = icmp slt i64 %tmp13, %n		; <i1> [#uses=1]
	br i1 %tmp14, label %bb, label %return.loopexit

return.loopexit:		; preds = %bb
	br label %return

return:		; preds = %return.loopexit, %entry
	ret void
}
define void @bar(i64 %n, i64 %m, i64 %o, i64 %q, double* nocapture %p) nounwind {
entry:
	%tmp = icmp sgt i64 %n, 0		; <i1> [#uses=1]
	br i1 %tmp, label %bb.nph, label %return

bb.nph:		; preds = %entry
	%tmp1 = mul i64 %n, %q		; <i64> [#uses=1]
	%tmp2 = mul i64 %tmp1, %m		; <i64> [#uses=1]
	%tmp3 = mul i64 %tmp2, %o		; <i64> [#uses=1]
	br label %bb

bb:		; preds = %bb, %bb.nph
	%i.01 = phi i64 [ %tmp3, %bb.nph ], [ %tmp13, %bb ]		; <i64> [#uses=3]
	%tmp9 = getelementptr double* %p, i64 %i.01		; <double*> [#uses=1]
	%tmp10 = load double* %tmp9, align 8		; <double> [#uses=1]
	%tmp11 = fdiv double %tmp10, 2.100000e+00		; <double> [#uses=1]
	store double %tmp11, double* %tmp9, align 8
	%tmp13 = add i64 %i.01, 1		; <i64> [#uses=2]
	%tmp14 = icmp slt i64 %tmp13, %n		; <i1> [#uses=1]
	br i1 %tmp14, label %bb, label %return.loopexit

return.loopexit:		; preds = %bb
	br label %return

return:		; preds = %return.loopexit, %entry
	ret void
}
