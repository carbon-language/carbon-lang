; RUN: llvm-as < %s | opt -loop-unswitch -disable-output
implementation   ; Functions:

int %test(int* %A, bool %C) {
entry:
	br label %no_exit

no_exit:		; preds = %entry, %no_exit.backedge
	%i.0.0 = phi uint [ 0, %entry ], [ %i.0.0.be, %no_exit.backedge ]		; <uint> [#uses=3]
	%tmp.7 = getelementptr int* %A, uint %i.0.0		; <int*> [#uses=4]
	%tmp.13 = load int* %tmp.7		; <int> [#uses=1]
	%tmp.14 = add int %tmp.13, 1		; <int> [#uses=1]
	store int %tmp.14, int* %tmp.7
	br bool %C, label %then, label %endif

then:		; preds = %no_exit
	%tmp.29 = load int* %tmp.7		; <int> [#uses=1]
	%tmp.30 = add int %tmp.29, 2		; <int> [#uses=1]
	store int %tmp.30, int* %tmp.7
	%inc9 = add uint %i.0.0, 1		; <uint> [#uses=2]
	%tmp.112 = setlt uint %inc9, 100000		; <bool> [#uses=1]
	br bool %tmp.112, label %no_exit.backedge, label %return

no_exit.backedge:		; preds = %then, %endif
	%i.0.0.be = phi uint [ %inc9, %then ], [ %inc, %endif ]		; <uint> [#uses=1]
	br label %no_exit

endif:		; preds = %no_exit
	%inc = add uint %i.0.0, 1		; <uint> [#uses=2]
	%tmp.1 = setlt uint %inc, 100000		; <bool> [#uses=1]
	br bool %tmp.1, label %no_exit.backedge, label %return

return:		; preds = %then, %endif
	ret int %tmp.13
}
