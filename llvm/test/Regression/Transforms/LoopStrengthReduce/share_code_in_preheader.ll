; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | grep mul | wc -l | grep 1
; LSR should not make two copies of the Q*L expression in the preheader!

sbyte %test(sbyte* %A, sbyte* %B, int %L, int %Q, int %N) {
entry:
	%tmp.6 = mul int %Q, %L		; <int> [#uses=1]
	%N = cast int %N to uint		; <uint> [#uses=1]
	br label %no_exit

no_exit:		; preds = %no_exit, %no_exit.preheader
	%indvar = phi uint [ 0, %entry], [ %indvar.next, %no_exit ]		; <uint> [#uses=2]
	%Sum.0.0 = phi sbyte [ 0, %entry], [ %tmp.21, %no_exit ]		; <sbyte> [#uses=1]
	%indvar = cast uint %indvar to int		; <int> [#uses=1]
	%N_addr.0.0 = sub int %N, %indvar		; <int> [#uses=1]
	%tmp.8 = add int %N_addr.0.0, %tmp.6		; <int> [#uses=2]
	%tmp.9 = getelementptr sbyte* %A, int %tmp.8		; <sbyte*> [#uses=1]
	%tmp.10 = load sbyte* %tmp.9		; <sbyte> [#uses=1]
	%tmp.17 = getelementptr sbyte* %B, int %tmp.8		; <sbyte*> [#uses=1]
	%tmp.18 = load sbyte* %tmp.17		; <sbyte> [#uses=1]
	%tmp.19 = sub sbyte %tmp.10, %tmp.18		; <sbyte> [#uses=1]
	%tmp.21 = add sbyte %tmp.19, %Sum.0.0		; <sbyte> [#uses=2]
	%indvar.next = add uint %indvar, 1		; <uint> [#uses=2]
	%exitcond = seteq uint %indvar.next, %N		; <bool> [#uses=1]
	br bool %exitcond, label %loopexit, label %no_exit

loopexit:
	ret sbyte %tmp.21
}
