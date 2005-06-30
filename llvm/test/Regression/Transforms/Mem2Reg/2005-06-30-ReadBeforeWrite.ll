; RUN: llvm-as < %s | opt -mem2reg -instcombine | llvm-dis | grep store
; PR590

void %zero(sbyte* %p, int %n) {
entry:
	%p_addr = alloca sbyte*		; <sbyte**> [#uses=2]
	%n_addr = alloca int		; <int*> [#uses=2]
	%i = alloca int		; <int*> [#uses=6]
	%out = alloca int		; <int*> [#uses=2]
	%undef = alloca int		; <int*> [#uses=2]
	store sbyte* %p, sbyte** %p_addr
	store int %n, int* %n_addr
	store int 0, int* %i
	br label %loopentry

loopentry:		; preds = %endif, %entry
	%tmp.0 = load int* %n_addr		; <int> [#uses=1]
	%tmp.1 = add int %tmp.0, 1		; <int> [#uses=1]
	%tmp.2 = load int* %i		; <int> [#uses=1]
	%tmp.3 = setgt int %tmp.1, %tmp.2		; <bool> [#uses=2]
	%tmp.4 = cast bool %tmp.3 to int		; <int> [#uses=0]
	br bool %tmp.3, label %no_exit, label %return

no_exit:		; preds = %loopentry
	%tmp.5 = load int* %undef		; <int> [#uses=1]
	store int %tmp.5, int* %out
	store int 0, int* %undef
	%tmp.6 = load int* %i		; <int> [#uses=1]
	%tmp.7 = setgt int %tmp.6, 0		; <bool> [#uses=2]
	%tmp.8 = cast bool %tmp.7 to int		; <int> [#uses=0]
	br bool %tmp.7, label %then, label %endif

then:		; preds = %no_exit
	%tmp.9 = load sbyte** %p_addr		; <sbyte*> [#uses=1]
	%tmp.10 = load int* %i		; <int> [#uses=1]
	%tmp.11 = sub int %tmp.10, 1		; <int> [#uses=1]
	%tmp.12 = getelementptr sbyte* %tmp.9, int %tmp.11		; <sbyte*> [#uses=1]
	%tmp.13 = load int* %out		; <int> [#uses=1]
	%tmp.14 = cast int %tmp.13 to sbyte		; <sbyte> [#uses=1]
	store sbyte %tmp.14, sbyte* %tmp.12
	br label %endif

endif:		; preds = %then, %no_exit
	%tmp.15 = load int* %i		; <int> [#uses=1]
	%inc = add int %tmp.15, 1		; <int> [#uses=1]
	store int %inc, int* %i
	br label %loopentry

return:		; preds = %loopentry
	ret void
}
