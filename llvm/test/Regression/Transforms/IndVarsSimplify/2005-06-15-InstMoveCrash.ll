; RUN: llvm-as < %s | opt -indvars -disable-output

void %main() {
entry:
	br label %no_exit.1.outer

no_exit.1.outer:		; preds = %endif.0, %entry
	%l_14237116.1.0.ph = phi sbyte [ -46, %entry ], [ 0, %endif.0 ]		; <sbyte> [#uses=1]
	%i.0.0.0.ph = phi int [ 0, %entry ], [ %inc.1, %endif.0 ]		; <int> [#uses=1]
	br label %no_exit.1

no_exit.1:		; preds = %_Z13func_47880058cc.exit, %no_exit.1.outer
	br bool false, label %_Z13func_47880058cc.exit, label %then.i

then.i:		; preds = %no_exit.1
	br label %_Z13func_47880058cc.exit

_Z13func_47880058cc.exit:		; preds = %then.i, %no_exit.1
	br bool false, label %then.0, label %no_exit.1

then.0:		; preds = %_Z13func_47880058cc.exit
	%tmp.6 = cast sbyte %l_14237116.1.0.ph to ubyte		; <ubyte> [#uses=1]
	br bool false, label %endif.0, label %then.1

then.1:		; preds = %then.0
	br label %endif.0

endif.0:		; preds = %then.1, %then.0
	%inc.1 = add int %i.0.0.0.ph, 1		; <int> [#uses=2]
	%tmp.2 = setgt int %inc.1, 99		; <bool> [#uses=1]
	br bool %tmp.2, label %loopexit.0, label %no_exit.1.outer

loopexit.0:		; preds = %endif.0
	%tmp.28 = cast ubyte %tmp.6 to uint		; <uint> [#uses=0]
	ret void
}
