; RUN: opt < %s -analyze -scalar-evolution -disable-output | grep { -->  {.*,+,.*}<%bb>} | count 8

; The addrecs in this loop are analyzable only by using nsw information.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"

define void @foo(double* %p) nounwind {
entry:
	%tmp = load double* %p, align 8		; <double> [#uses=1]
	%tmp1 = fcmp ogt double %tmp, 2.000000e+00		; <i1> [#uses=1]
	br i1 %tmp1, label %bb.nph, label %return

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb1, %bb.nph
	%i.01 = phi i32 [ %tmp8, %bb1 ], [ 0, %bb.nph ]		; <i32> [#uses=3]
	%tmp2 = sext i32 %i.01 to i64		; <i64> [#uses=1]
	%tmp3 = getelementptr double* %p, i64 %tmp2		; <double*> [#uses=1]
	%tmp4 = load double* %tmp3, align 8		; <double> [#uses=1]
	%tmp5 = fmul double %tmp4, 9.200000e+00		; <double> [#uses=1]
	%tmp6 = sext i32 %i.01 to i64		; <i64> [#uses=1]
	%tmp7 = getelementptr double* %p, i64 %tmp6		; <double*> [#uses=1]
	store double %tmp5, double* %tmp7, align 8
	%tmp8 = add nsw i32 %i.01, 1		; <i32> [#uses=2]
	br label %bb1

bb1:		; preds = %bb
	%phitmp = sext i32 %tmp8 to i64		; <i64> [#uses=1]
	%tmp9 = getelementptr double* %p, i64 %phitmp		; <double*> [#uses=1]
	%tmp10 = load double* %tmp9, align 8		; <double> [#uses=1]
	%tmp11 = fcmp ogt double %tmp10, 2.000000e+00		; <i1> [#uses=1]
	br i1 %tmp11, label %bb, label %bb1.return_crit_edge

bb1.return_crit_edge:		; preds = %bb1
	br label %return

return:		; preds = %bb1.return_crit_edge, %entry
	ret void
}
