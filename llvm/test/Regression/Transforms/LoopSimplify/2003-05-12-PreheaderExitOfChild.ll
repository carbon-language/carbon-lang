; This (complex) testcase causes an assertion failure because a preheader is 
; inserted for the "fail" loop, but the exit block of a loop is not updated
; to be the preheader instead of the exit loop itself.

; RUN: as < %s | opt -preheaders

int %re_match_2() {
	br label %loopentry.1

loopentry.1:		; preds = %then.6, %endif.7, %loopexit.20, %endif.83
	br label %shortcirc_done.36

shortcirc_done.36:		; preds = %label.13, %shortcirc_next.36
	br bool false, label %fail, label %endif.40

endif.40:		; preds = %shortcirc_done.36
	br label %loopexit.20

loopentry.20:		; preds = %shortcirc_done.40, %endif.46
	br label %loopexit.20

loopexit.20:		; preds = %loopentry.20
	br label %loopentry.21

loopentry.21:		; preds = %loopexit.20, %no_exit.19
	br bool false, label %no_exit.19, label %loopexit.21

no_exit.19:		; preds = %loopentry.21
	br bool false, label %fail, label %loopentry.21

loopexit.21:		; preds = %loopentry.21
	br label %endif.45

endif.45:		; preds = %loopexit.21
	br label %cond_true.15

cond_true.15:		; preds = %endif.45
	br bool false, label %fail, label %endif.46

endif.46:		; preds = %cond_true.15
	br label %loopentry.20

fail:		; preds = %shortcirc_done.36, %loopexit.37, %cond_true.15, %no_exit.19
	br label %then.80

then.80:		; preds = %fail
	br label %endif.81

endif.81:		; preds = %then.80
	br label %loopexit.37

loopexit.37:		; preds = %endif.81
	br bool false, label %fail, label %endif.82

endif.82:		; preds = %loopexit.37
	br label %loopentry.1
}
