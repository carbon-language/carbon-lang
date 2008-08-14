; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | grep "phi double"

define void @foobar(i32 %n) nounwind {
entry:
	icmp eq i32 %n, 0		; <i1>:0 [#uses=2]
	br i1 %0, label %return, label %bb.nph

bb.nph:		; preds = %entry
	%umax = select i1 %0, i32 1, i32 %n		; <i32> [#uses=1]
	br label %bb

bb:		; preds = %bb, %bb.nph
	%i.03 = phi i32 [ 0, %bb.nph ], [ %indvar.next, %bb ]		; <i32> [#uses=3]
	tail call void @bar( i32 %i.03 ) nounwind
	uitofp i32 %i.03 to double		; <double>:1 [#uses=1]
	tail call void @foo( double %1 ) nounwind
	%indvar.next = add i32 %i.03, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %umax		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb, %entry
	ret void
}

declare void @bar(i32)

declare void @foo(double)
