; RUN: llvm-as < %s | opt -simplifycfg -disable-output


void %NewExtractNames() {
entry:
	br bool false, label %endif.0, label %then.0

then.0:		; preds = %entry
	br bool false, label %shortcirc_next.i, label %shortcirc_done.i

shortcirc_next.i:		; preds = %then.0
	br label %shortcirc_done.i

shortcirc_done.i:		; preds = %shortcirc_next.i, %then.0
	br bool false, label %then.0.i, label %else.0.i

then.0.i:		; preds = %shortcirc_done.i
	br label %NewBase.exit

else.0.i:		; preds = %shortcirc_done.i
	br bool false, label %endif.0.i, label %else.1.i

else.1.i:		; preds = %else.0.i
	br bool false, label %endif.0.i, label %else.2.i

else.2.i:		; preds = %else.1.i
	br label %NewBase.exit

endif.0.i:		; preds = %else.1.i, %else.0.i
	br label %NewBase.exit

NewBase.exit:		; preds = %endif.0.i, %else.2.i, %then.0.i
	br label %endif.0

endif.0:		; preds = %NewBase.exit, %entry
	%tmp.32.mask = and uint 0, 31		; <uint> [#uses=1]
	switch uint %tmp.32.mask, label %label.9 [
		 uint 16, label %loopentry.2
		 uint 15, label %loopentry.2
		 uint 14, label %loopentry.2
		 uint 13, label %loopentry.2
		 uint 10, label %loopentry.2
		 uint 20, label %loopentry.1
		 uint 19, label %loopentry.1
		 uint 2, label %loopentry.0
		 uint 0, label %switchexit
	]

loopentry.0:		; preds = %endif.1, %endif.0
	br bool false, label %no_exit.0, label %switchexit

no_exit.0:		; preds = %loopentry.0
	br bool false, label %then.1, label %else.1

then.1:		; preds = %no_exit.0
	br label %endif.1

else.1:		; preds = %no_exit.0
	br bool false, label %shortcirc_next.0, label %shortcirc_done.0

shortcirc_next.0:		; preds = %else.1
	br label %shortcirc_done.0

shortcirc_done.0:		; preds = %shortcirc_next.0, %else.1
	br bool false, label %then.2, label %endif.2

then.2:		; preds = %shortcirc_done.0
	br label %endif.2

endif.2:		; preds = %then.2, %shortcirc_done.0
	br label %endif.1

endif.1:		; preds = %endif.2, %then.1
	br label %loopentry.0

loopentry.1:		; preds = %endif.3, %endif.0, %endif.0
	br bool false, label %no_exit.1, label %switchexit

no_exit.1:		; preds = %loopentry.1
	br bool false, label %then.3, label %else.2

then.3:		; preds = %no_exit.1
	br label %endif.3

else.2:		; preds = %no_exit.1
	br bool false, label %shortcirc_next.1, label %shortcirc_done.1

shortcirc_next.1:		; preds = %else.2
	br label %shortcirc_done.1

shortcirc_done.1:		; preds = %shortcirc_next.1, %else.2
	br bool false, label %then.4, label %endif.4

then.4:		; preds = %shortcirc_done.1
	br label %endif.4

endif.4:		; preds = %then.4, %shortcirc_done.1
	br label %endif.3

endif.3:		; preds = %endif.4, %then.3
	br label %loopentry.1

loopentry.2:		; preds = %endif.5, %endif.0, %endif.0, %endif.0, %endif.0, %endif.0
	%i.3 = phi int [ 0, %endif.5 ], [ 0, %endif.0 ], [ 0, %endif.0 ], [ 0, %endif.0 ], [ 0, %endif.0 ], [ 0, %endif.0 ]		; <int> [#uses=1]
	%tmp.158 = setlt int %i.3, 0		; <bool> [#uses=1]
	br bool %tmp.158, label %no_exit.2, label %switchexit

no_exit.2:		; preds = %loopentry.2
	br bool false, label %shortcirc_next.2, label %shortcirc_done.2

shortcirc_next.2:		; preds = %no_exit.2
	br label %shortcirc_done.2

shortcirc_done.2:		; preds = %shortcirc_next.2, %no_exit.2
	br bool false, label %then.5, label %endif.5

then.5:		; preds = %shortcirc_done.2
	br label %endif.5

endif.5:		; preds = %then.5, %shortcirc_done.2
	br label %loopentry.2

label.9:		; preds = %endif.0
	br bool false, label %then.6, label %endif.6

then.6:		; preds = %label.9
	br label %endif.6

endif.6:		; preds = %then.6, %label.9
	store int 0, int* null
	br label %switchexit

switchexit:		; preds = %endif.6, %loopentry.2, %loopentry.1, %loopentry.0, %endif.0
	br bool false, label %endif.7, label %then.7

then.7:		; preds = %switchexit
	br bool false, label %shortcirc_next.3, label %shortcirc_done.3

shortcirc_next.3:		; preds = %then.7
	br label %shortcirc_done.3

shortcirc_done.3:		; preds = %shortcirc_next.3, %then.7
	br bool false, label %then.8, label %endif.8

then.8:		; preds = %shortcirc_done.3
	br label %endif.8

endif.8:		; preds = %then.8, %shortcirc_done.3
	br label %endif.7

endif.7:		; preds = %endif.8, %switchexit
	ret void
}
