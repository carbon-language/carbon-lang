; RUN: llvm-as < %s | opt -loop-unswitch -disable-output

void %sort_Eq(int * %S2) {
entry:
	br bool false, label %list_Length.exit, label %cond_true.i

cond_true.i:		; preds = %entry
	ret void

list_Length.exit:		; preds = %entry
	br bool false, label %list_Length.exit9, label %cond_true.i5

cond_true.i5:		; preds = %list_Length.exit
	ret void

list_Length.exit9:		; preds = %list_Length.exit
	br bool false, label %bb78, label %return

bb44:		; preds = %bb78, %cond_next68
	br bool %tmp49.not, label %bb62, label %bb62.loopexit

bb62.loopexit:		; preds = %bb44
	br label %bb62

bb62:		; preds = %bb62.loopexit, %bb44
	br bool false, label %return.loopexit, label %cond_next68

cond_next68:		; preds = %bb62
	br bool false, label %return.loopexit, label %bb44

bb78:		; preds = %list_Length.exit9
	%tmp49.not = seteq int* %S2, null		; <bool> [#uses=1]
	br label %bb44

return.loopexit:		; preds = %cond_next68, %bb62
	%retval.0.ph = phi uint [ 1, %cond_next68 ], [ 0, %bb62 ]		; <uint> [#uses=1]
	br label %return

return:		; preds = %return.loopexit, %list_Length.exit9
	%retval.0 = phi uint [ 0, %list_Length.exit9 ], [ %retval.0.ph, %return.loopexit ]		; <uint> [#uses=0]
	ret void
}
