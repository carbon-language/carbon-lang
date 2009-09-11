; RUN: opt < %s -indvars -S > %t
; RUN: grep {%exitcond = icmp eq i64 %indvar.next, %n} %t
; RUN: grep {getelementptr i8\\* %A, i64 %indvar} %t
; RUN: grep getelementptr %t | count 1
; RUN: grep add %t | count 1
; RUN: not grep scevgep %t
; RUN: not grep ptrtoint %t

; Indvars should be able to expand the pointer-arithmetic
; IV into an integer IV indexing into a simple getelementptr.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"

define void @foo(i8* %A, i64 %n) nounwind {
entry:
	%0 = icmp eq i64 %n, 0		; <i1> [#uses=1]
	br i1 %0, label %return, label %bb.nph

bb.nph:		; preds = %entry
	%1 = getelementptr i8* %A, i64 %n		; <i8*> [#uses=1]
	br label %bb

bb:		; preds = %bb1, %bb.nph
	%q.01 = phi i8* [ %2, %bb1 ], [ %A, %bb.nph ]		; <i8*> [#uses=2]
	store i8 0, i8* %q.01, align 1
	%2 = getelementptr i8* %q.01, i64 1		; <i8*> [#uses=2]
	br label %bb1

bb1:		; preds = %bb
	%3 = icmp eq i8* %1, %2		; <i1> [#uses=1]
	br i1 %3, label %bb1.return_crit_edge, label %bb

bb1.return_crit_edge:		; preds = %bb1
	br label %return

return:		; preds = %bb1.return_crit_edge, %entry
	ret void
}
