; RUN: llvm-as < %s | opt -loop-extract -disable-output

void %sendMTFValues() {
entry:
	br bool false, label %then.1, label %endif.1

then.1:		; preds = %entry
	br bool false, label %loopentry.6.preheader, label %else.0

endif.1:		; preds = %entry
	ret void

else.0:		; preds = %then.1
	ret void

loopentry.6.preheader:		; preds = %then.1
	br bool false, label %endif.7.preheader, label %loopexit.9

endif.7.preheader:		; preds = %loopentry.6.preheader
	%tmp.183 = add int 0, -1		; <int> [#uses=1]
	br label %endif.7

endif.7:		; preds = %endif.7.preheader, %loopexit.15
	br bool false, label %loopentry.10, label %loopentry.12

loopentry.10:		; preds = %endif.7
	br label %loopentry.12

loopentry.12:		; preds = %endif.7, %loopentry.10
	%ge.2.1 = phi int [ 0, %loopentry.10 ], [ %tmp.183, %endif.7 ]		; <int> [#uses=0]
	br bool false, label %loopexit.14, label %no_exit.11

no_exit.11:		; preds = %loopentry.12
	ret void

loopexit.14:		; preds = %loopentry.12
	br bool false, label %loopexit.15, label %no_exit.14

no_exit.14:		; preds = %loopexit.14
	ret void

loopexit.15:		; preds = %loopexit.14
	br bool false, label %endif.7, label %loopexit.9

loopexit.9:		; preds = %loopentry.6.preheader, %loopexit.15
	ret void
}
