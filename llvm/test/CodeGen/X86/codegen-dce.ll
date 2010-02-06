; RUN: llc < %s -march=x86 -stats |& grep {codegen-dce} | grep {Number of dead instructions deleted}

	%struct.anon = type { [3 x double], double, %struct.node*, [64 x %struct.bnode*], [64 x %struct.bnode*] }
	%struct.bnode = type { i16, double, [3 x double], i32, i32, [3 x double], [3 x double], [3 x double], double, %struct.bnode*, %struct.bnode* }
	%struct.node = type { i16, double, [3 x double], i32, i32 }

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind {
entry:
	%0 = malloc %struct.anon		; <%struct.anon*> [#uses=2]
	%1 = getelementptr %struct.anon* %0, i32 0, i32 2		; <%struct.node**> [#uses=1]
	br label %bb14.i

bb14.i:		; preds = %bb14.i, %entry
	%i8.0.reg2mem.0.i = phi i32 [ 0, %entry ], [ %2, %bb14.i ]		; <i32> [#uses=1]
	%2 = add i32 %i8.0.reg2mem.0.i, 1		; <i32> [#uses=2]
	%exitcond74.i = icmp eq i32 %2, 32		; <i1> [#uses=1]
	br i1 %exitcond74.i, label %bb32.i, label %bb14.i

bb32.i:		; preds = %bb32.i, %bb14.i
	%tmp.0.reg2mem.0.i = phi i32 [ %indvar.next63.i, %bb32.i ], [ 0, %bb14.i ]		; <i32> [#uses=1]
	%indvar.next63.i = add i32 %tmp.0.reg2mem.0.i, 1		; <i32> [#uses=2]
	%exitcond64.i = icmp eq i32 %indvar.next63.i, 64		; <i1> [#uses=1]
	br i1 %exitcond64.i, label %bb47.loopexit.i, label %bb32.i

bb.i.i:		; preds = %bb47.loopexit.i
	unreachable

stepsystem.exit.i:		; preds = %bb47.loopexit.i
	store %struct.node* null, %struct.node** %1, align 4
	br label %bb.i6.i

bb.i6.i:		; preds = %bb.i6.i, %stepsystem.exit.i
	br i1 false, label %bb107.i.i, label %bb.i6.i

bb107.i.i:		; preds = %bb107.i.i, %bb.i6.i
	%q_addr.0.i.i.in = phi %struct.bnode** [ null, %bb107.i.i ], [ %3, %bb.i6.i ]		; <%struct.bnode**> [#uses=0]
	br label %bb107.i.i

bb47.loopexit.i:		; preds = %bb32.i
	%3 = getelementptr %struct.anon* %0, i32 0, i32 4, i32 0		; <%struct.bnode**> [#uses=1]
	%4 = icmp eq %struct.node* null, null		; <i1> [#uses=1]
	br i1 %4, label %stepsystem.exit.i, label %bb.i.i
}
