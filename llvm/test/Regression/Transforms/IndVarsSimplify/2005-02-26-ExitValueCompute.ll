; RUN: llvm-as < %s | opt -indvars | llvm-dis | grep 'ret int 152'

int %main() {
entry:
	br label %no_exit

no_exit:		; preds = %no_exit, %entry
	%i.1.0 = phi int [ 0, %entry ], [ %inc, %no_exit ]		; <int> [#uses=2]
	%tmp.4 = setgt int %i.1.0, 50		; <bool> [#uses=1]
	%tmp.7 = select bool %tmp.4, int 100, int 0		; <int> [#uses=1]
	%i.0 = add int %i.1.0, 1		; <int> [#uses=1]
	%inc = add int %i.0, %tmp.7		; <int> [#uses=3]
	%tmp.1 = setlt int %inc, 100		; <bool> [#uses=1]
	br bool %tmp.1, label %no_exit, label %loopexit

loopexit:		; preds = %no_exit
	ret int %inc
}

