; RUN: llvm-as < %s | opt -indvars -disable-output

implementation   ; Functions:

declare void %q_atomic_increment()

declare void %_Z9qt_assertPKcS0_i()

void %_ZN13QMetaResourceC1EPKh() {
entry:
	invoke void %_Z9qt_assertPKcS0_i( )
			to label %endif.1 unwind label %then.i.i551

then.i.i551:		; preds = %entry
	ret void

endif.1:		; preds = %entry
	br bool false, label %then.2, label %then.i.i

then.2:		; preds = %endif.1
	invoke void %q_atomic_increment( )
			to label %loopentry.0 unwind label %invoke_catch.6

invoke_catch.6:		; preds = %then.2
	ret void

loopentry.0:		; preds = %then.2
	br bool false, label %shortcirc_next.i, label %endif.3

endif.3:		; preds = %loopentry.0
	ret void

shortcirc_next.i:		; preds = %loopentry.0
	br bool false, label %_ZNK7QString2atEi.exit, label %then.i

then.i:		; preds = %shortcirc_next.i
	ret void

_ZNK7QString2atEi.exit:		; preds = %shortcirc_next.i
	br bool false, label %endif.4, label %then.4

then.4:		; preds = %_ZNK7QString2atEi.exit
	ret void

endif.4:		; preds = %_ZNK7QString2atEi.exit
	%tmp.115 = load ubyte* null		; <ubyte> [#uses=1]
	br bool false, label %loopexit.1, label %no_exit.0

no_exit.0:		; preds = %no_exit.0, %endif.4
	%bytes_in_len.4.5 = phi ubyte [ %dec, %no_exit.0 ], [ %tmp.115, %endif.4 ]		; <ubyte> [#uses=1]
	%off.5.5.in = phi int [ %off.5.5, %no_exit.0 ], [ 0, %endif.4 ]		; <int> [#uses=1]
	%off.5.5 = add int %off.5.5.in, 1		; <int> [#uses=2]
	%dec = add ubyte %bytes_in_len.4.5, 255		; <ubyte> [#uses=2]
	%tmp.123631 = seteq ubyte %dec, 0		; <bool> [#uses=1]
	br bool %tmp.123631, label %loopexit.1, label %no_exit.0

loopexit.1:		; preds = %no_exit.0, %endif.4
	%off.5.in.6 = phi int [ 0, %endif.4 ], [ %off.5.5, %no_exit.0 ]		; <int> [#uses=0]
	ret void

then.i.i:		; preds = %endif.1
	ret void
}
