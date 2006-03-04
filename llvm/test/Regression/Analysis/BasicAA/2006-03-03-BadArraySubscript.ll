; RUN: llvm-as < %s | opt -aa-eval -disable-output 2>&1 | grep '2 no alias respon'

;; TEST that A[1][0] may alias A[0][i].

void %test(int %N) {
entry:
	%X = alloca [3 x [3 x int]]		; <[3 x [3 x int]]*> [#uses=4]
	%tmp.24 = setgt int %N, 0		; <bool> [#uses=1]
	br bool %tmp.24, label %no_exit, label %loopexit

no_exit:		; preds = %no_exit, %entry
	%i.0.0 = phi int [ 0, %entry ], [ %inc, %no_exit ]		; <int> [#uses=2]
	%tmp.6 = getelementptr [3 x [3 x int]]* %X, int 0, int 0, int %i.0.0		; <int*> [#uses=1]
	store int 1, int* %tmp.6
	%tmp.8 = getelementptr [3 x [3 x int]]* %X, int 0, int 0, int 0		; <int*> [#uses=1]
	%tmp.9 = load int* %tmp.8		; <int> [#uses=1]
	%tmp.11 = getelementptr [3 x [3 x int]]* %X, int 0, int 1, int 0		; <int*> [#uses=1]
	%tmp.12 = load int* %tmp.11		; <int> [#uses=1]
	%tmp.13 = add int %tmp.12, %tmp.9		; <int> [#uses=1]
	%inc = add int %i.0.0, 1		; <int> [#uses=2]
	%tmp.2 = setlt int %inc, %N		; <bool> [#uses=1]
	br bool %tmp.2, label %no_exit, label %loopexit

loopexit:		; preds = %no_exit, %entry
	%Y.0.1 = phi int [ 0, %entry ], [ %tmp.13, %no_exit ]		; <int> [#uses=1]
	%tmp.4 = getelementptr [3 x [3 x int]]* %X, int 0, int 0		; <[3 x int]*> [#uses=1]
	%tmp.15 = call int (...)* %foo( [3 x int]* %tmp.4, int %Y.0.1 )		; <int> [#uses=0]
	ret void
}

declare int %foo(...)
