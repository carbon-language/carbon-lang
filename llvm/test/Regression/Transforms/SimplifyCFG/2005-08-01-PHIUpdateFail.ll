; RUN: llvm-as < %s | opt -simplifycfg -disable-output
; XFAIL: *

void %main() {
entry:
	%tmp.14.i19 = seteq int 0, 2		; <bool> [#uses=1]
	br bool %tmp.14.i19, label %endif.1.i20, label %read_min.exit

endif.1.i20:		; preds = %entry
	%tmp.9.i.i = seteq sbyte* null, null		; <bool> [#uses=1]
	br bool %tmp.9.i.i, label %then.i12.i, label %then.i.i

then.i.i:		; preds = %endif.1.i20
	ret void

then.i12.i:		; preds = %endif.1.i20
	%tmp.9.i4.i = seteq sbyte* null, null		; <bool> [#uses=1]
	br bool %tmp.9.i4.i, label %endif.2.i33, label %then.i5.i

then.i5.i:		; preds = %then.i12.i
	ret void

endif.2.i33:		; preds = %then.i12.i
	br bool false, label %loopexit.0.i40, label %no_exit.0.i35

no_exit.0.i35:		; preds = %no_exit.0.i35, %endif.2.i33
	%tmp.130.i = setlt int 0, 0		; <bool> [#uses=1]
	br bool %tmp.130.i, label %loopexit.0.i40.loopexit, label %no_exit.0.i35

loopexit.0.i40.loopexit:		; preds = %no_exit.0.i35
	br label %loopexit.0.i40

loopexit.0.i40:		; preds = %loopexit.0.i40.loopexit, %endif.2.i33
	%tmp.341.i = seteq int 0, 0		; <bool> [#uses=1]
	br bool %tmp.341.i, label %loopentry.1.i, label %read_min.exit

loopentry.1.i:		; preds = %loopexit.0.i40
	%tmp.347.i = setgt int 0, 0		; <bool> [#uses=1]
	br bool %tmp.347.i, label %no_exit.1.i41, label %loopexit.2.i44

no_exit.1.i41:		; preds = %endif.5.i, %loopentry.1.i
	%indvar.i42 = phi uint [ %indvar.next.i, %endif.5.i ], [ 0, %loopentry.1.i ]		; <uint> [#uses=1]
	%tmp.355.i = seteq int 0, 3		; <bool> [#uses=1]
	br bool %tmp.355.i, label %endif.5.i, label %read_min.exit

endif.5.i:		; preds = %no_exit.1.i41
	%tmp.34773.i = setgt int 0, 0		; <bool> [#uses=1]
	%indvar.next.i = add uint %indvar.i42, 1		; <uint> [#uses=1]
	br bool %tmp.34773.i, label %no_exit.1.i41, label %loopexit.1.i.loopexit

loopexit.1.i.loopexit:		; preds = %endif.5.i
	ret void

loopexit.2.i44:		; preds = %loopentry.1.i
	ret void

read_min.exit:		; preds = %no_exit.1.i41, %loopexit.0.i40, %entry
	%tmp.23 = seteq int 0, 0		; <bool> [#uses=1]
	br bool %tmp.23, label %endif.1, label %then.1

then.1:		; preds = %read_min.exit
	br bool false, label %endif.0.i, label %then.0.i

then.0.i:		; preds = %then.1
	br bool false, label %endif.1.i, label %then.1.i

endif.0.i:		; preds = %then.1
	br bool false, label %endif.1.i, label %then.1.i

then.1.i:		; preds = %endif.0.i, %then.0.i
	br bool false, label %getfree.exit, label %then.2.i

endif.1.i:		; preds = %endif.0.i, %then.0.i
	br bool false, label %getfree.exit, label %then.2.i

then.2.i:		; preds = %endif.1.i, %then.1.i
	ret void

getfree.exit:		; preds = %endif.1.i, %then.1.i
	ret void

endif.1:		; preds = %read_min.exit
	%tmp.27.i = getelementptr int* null, int 0
	br bool false, label %loopexit.0.i15, label %no_exit.0.i14

no_exit.0.i14:		; preds = %endif.1
	ret void

loopexit.0.i15:		; preds = %endif.1
	br bool false, label %primal_start_artificial.exit, label %no_exit.1.i16

no_exit.1.i16:		; preds = %no_exit.1.i16, %loopexit.0.i15
	br bool false, label %primal_start_artificial.exit, label %no_exit.1.i16

primal_start_artificial.exit:		; preds = %no_exit.1.i16, %loopexit.0.i15
	ret void
}
