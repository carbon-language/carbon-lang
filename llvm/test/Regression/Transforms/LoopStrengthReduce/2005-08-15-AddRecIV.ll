; RUN: llvm-as < %s | opt -loop-reduce -disable-output

void %try_swap() {
entry:
	br bool false, label %cond_continue.0.i, label %cond_false.0.i

cond_false.0.i:		; preds = %entry
	ret void

cond_continue.0.i:		; preds = %entry
	br bool false, label %cond_continue.1.i, label %cond_false.1.i

cond_false.1.i:		; preds = %cond_continue.0.i
	ret void

cond_continue.1.i:		; preds = %cond_continue.0.i
	br bool false, label %endif.3.i, label %else.0.i

endif.3.i:		; preds = %cond_continue.1.i
	br bool false, label %my_irand.exit82, label %endif.0.i62

else.0.i:		; preds = %cond_continue.1.i
	ret void

endif.0.i62:		; preds = %endif.3.i
	ret void

my_irand.exit82:		; preds = %endif.3.i
	br bool false, label %else.2, label %then.4

then.4:		; preds = %my_irand.exit82
	ret void

else.2:		; preds = %my_irand.exit82
	br bool false, label %find_affected_nets.exit, label %loopentry.1.i107.outer.preheader

loopentry.1.i107.outer.preheader:		; preds = %else.2
	ret void

find_affected_nets.exit:		; preds = %else.2
	br bool false, label %save_region_occ.exit, label %loopentry.1

save_region_occ.exit:		; preds = %find_affected_nets.exit
	br bool false, label %no_exit.1.preheader, label %loopexit.1

loopentry.1:		; preds = %find_affected_nets.exit
	ret void

no_exit.1.preheader:		; preds = %save_region_occ.exit
	ret void

loopexit.1:		; preds = %save_region_occ.exit
	br bool false, label %then.10, label %loopentry.3

then.10:		; preds = %loopexit.1
	ret void

loopentry.3:		; preds = %endif.16, %loopexit.1
	%indvar342 = phi uint [ %indvar.next343, %endif.16 ], [ 0, %loopexit.1 ]		; <uint> [#uses=2]
	br bool false, label %loopexit.3, label %endif.16

endif.16:		; preds = %loopentry.3
	%indvar.next343 = add uint %indvar342, 1		; <uint> [#uses=1]
	br label %loopentry.3

loopexit.3:		; preds = %loopentry.3
	br label %loopentry.4

loopentry.4:		; preds = %loopentry.4, %loopexit.3
	%indvar340 = phi uint [ 0, %loopexit.3 ], [ %indvar.next341, %loopentry.4 ]		; <uint> [#uses=2]
	%tmp. = add uint %indvar340, %indvar342		; <uint> [#uses=1]
	%tmp.526 = load int** null		; <int*> [#uses=1]
	%tmp.528 = getelementptr int* %tmp.526, uint %tmp.		; <int*> [#uses=1]
	store int 0, int* %tmp.528
	%indvar.next341 = add uint %indvar340, 1		; <uint> [#uses=1]
	br label %loopentry.4
}
