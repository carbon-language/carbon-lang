; RUN: llvm-as < %s | opt -simplifycfg -disable-output

implementation   ; Functions:

void %symhash_add() {
entry:
	br bool undef, label %then.0, label %UnifiedReturnBlock

then.0:		; preds = %entry
	br bool undef, label %loopentry.2, label %loopentry.1.preheader

loopentry.1.preheader:		; preds = %then.0
	br label %loopentry.1.outer

loopentry.1.outer:		; preds = %loopentry.1.preheader, %loopexit.1
	br label %loopentry.1

loopentry.1:		; preds = %loopentry.1.outer, %then.1, %then.3, %then.4, %endif.1
	br bool undef, label %loopexit.1, label %no_exit.1

no_exit.1:		; preds = %loopentry.1
	br bool undef, label %then.1, label %else.0

then.1:		; preds = %no_exit.1
	br label %loopentry.1

else.0:		; preds = %no_exit.1
	br bool undef, label %then.2, label %else.1

then.2:		; preds = %else.0
	br bool undef, label %then.3, label %endif.1

then.3:		; preds = %then.2
	br label %loopentry.1

else.1:		; preds = %else.0
	br bool undef, label %endif.1, label %then.4

then.4:		; preds = %else.1
	br label %loopentry.1

endif.1:		; preds = %then.2, %else.1
	br label %loopentry.1

loopexit.1:		; preds = %loopentry.1
	br bool undef, label %loopentry.1.outer, label %loopentry.2

loopentry.2:		; preds = %then.0, %loopexit.1, %no_exit.2
	br bool undef, label %loopexit.2, label %no_exit.2

no_exit.2:		; preds = %loopentry.2
	br label %loopentry.2

loopexit.2:		; preds = %loopentry.2
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}
