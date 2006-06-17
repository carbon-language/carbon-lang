; RUN: llvm-as < %s | opt -indvars -disable-output

void %get_block() {
endif.0:		; preds = %entry
	br label %no_exit.30

no_exit.30:		; preds = %no_exit.30, %endif.0
	%x.12.0 = phi int [ %inc.28, %no_exit.30 ], [ -2, %endif.0 ]		; <int> [#uses=1]
	%tmp.583 = load ushort* null		; <ushort> [#uses=1]
	%tmp.584 = cast ushort %tmp.583 to int		; <int> [#uses=1]
	%tmp.588 = load int* null		; <int> [#uses=1]
	%tmp.589 = mul int %tmp.584, %tmp.588		; <int> [#uses=1]
	%tmp.591 = add int %tmp.589, 0		; <int> [#uses=1]
	%inc.28 = add int %x.12.0, 1		; <int> [#uses=2]
	%tmp.565 = setgt int %inc.28, 3		; <bool> [#uses=1]
	br bool %tmp.565, label %loopexit.30, label %no_exit.30

loopexit.30:		; preds = %no_exit.30
	%tmp.591.lcssa = phi int [ %tmp.591, %no_exit.30 ]		; <int> [#uses=0]
	ret void
}
