; RUN: if as < %s | opt -adce -simplifycfg | dis | grep then:
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

void %dead_test8(int* %data.1, int %idx.1) {
entry:		; No predecessors!
	%tmp.1 = load int* %data.1		; <int> [#uses=2]
	%tmp.41 = setgt int %tmp.1, 0		; <bool> [#uses=1]
	br bool %tmp.41, label %no_exit.preheader, label %return

no_exit.preheader:		; preds = %entry
	%tmp.11 = getelementptr int* %data.1, long 1		; <int*> [#uses=1]
	%tmp.22-idxcast = cast int %idx.1 to long		; <long> [#uses=1]
	%tmp.28 = getelementptr int* %data.1, long %tmp.22-idxcast		; <int*> [#uses=1]
	br label %no_exit

no_exit:		; preds = %no_exit.preheader, %endif
	%k.1 = phi int [ %k.0, %endif ], [ 0, %no_exit.preheader ]		; <int> [#uses=3]
	%i.0 = phi int [ %inc.1, %endif ], [ 0, %no_exit.preheader ]		; <int> [#uses=1]
	%tmp.12 = load int* %tmp.11		; <int> [#uses=1]
	%tmp.14 = sub int 0, %tmp.12		; <int> [#uses=1]
	%tmp.161 = setne int %k.1, %tmp.14		; <bool> [#uses=1]
	br bool %tmp.161, label %then, label %else

then:		; preds = %no_exit
	%inc.0 = add int %k.1, 1		; <int> [#uses=1]
	br label %endif

else:		; preds = %no_exit
	%dec = add int %k.1, -1		; <int> [#uses=1]
	br label %endif

endif:		; preds = %else, %then
	%k.0 = phi int [ %dec, %else ], [ %inc.0, %then ]		; <int> [#uses=1]
	store int 2, int* %tmp.28
	%inc.1 = add int %i.0, 1		; <int> [#uses=2]
	%tmp.4 = setlt int %inc.1, %tmp.1		; <bool> [#uses=1]
	br bool %tmp.4, label %no_exit, label %return

return:		; preds = %entry, %endif
	ret void
}
