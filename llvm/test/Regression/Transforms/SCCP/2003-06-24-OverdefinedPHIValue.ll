; RUN: as < %s | opt -sccp -simplifycfg | dis | not grep then:

void %cprop_test11(int* %data.1) {
entry:		; No predecessors!
	%tmp.1 = load int* %data.1		; <int> [#uses=3]
	%tmp.41 = setgt int %tmp.1, 1		; <bool> [#uses=1]
	br bool %tmp.41, label %no_exit, label %loopexit

no_exit:		; preds = %entry, %then, %endif
	%j.0 = phi int [ %j.0, %endif ], [ %i.0, %then ], [ 1, %entry ]		; <int> [#uses=3]
	%i.0 = phi int [ %inc, %endif ], [ %inc1, %then ], [ 1, %entry ]		; <int> [#uses=4]
	%tmp.8.not = cast int %j.0 to bool		; <bool> [#uses=1]
	br bool %tmp.8.not, label %endif, label %then

then:		; preds = %no_exit
	%inc1 = add int %i.0, 1		; <int> [#uses=3]
	%tmp.42 = setlt int %inc1, %tmp.1		; <bool> [#uses=1]
	br bool %tmp.42, label %no_exit, label %loopexit

endif:		; preds = %no_exit
	%inc = add int %i.0, 1		; <int> [#uses=3]
	%tmp.4 = setlt int %inc, %tmp.1		; <bool> [#uses=1]
	br bool %tmp.4, label %no_exit, label %loopexit

loopexit:		; preds = %entry, %endif, %then
	%j.1 = phi int [ 1, %entry ], [ %j.0, %endif ], [ %i.0, %then ]		; <int> [#uses=1]
	%i.1 = phi int [ 1, %entry ], [ %inc, %endif ], [ %inc1, %then ]		; <int> [#uses=1]
	%tmp.17 = getelementptr int* %data.1, long 1		; <int*> [#uses=1]
	store int %j.1, int* %tmp.17
	%tmp.23 = getelementptr int* %data.1, long 2		; <int*> [#uses=1]
	store int %i.1, int* %tmp.23
	ret void
}
