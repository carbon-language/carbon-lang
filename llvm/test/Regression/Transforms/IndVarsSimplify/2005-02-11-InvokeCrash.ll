; RUN: llvm-as < %s | opt -indvars -disable-output

void %_ZN5ArrayISt7complexIdEEC2ERK10dim_vector() {
entry:
	%tmp.7 = invoke int %_ZN5ArrayISt7complexIdEE8get_sizeERK10dim_vector( )
			to label %invoke_cont.0 unwind label %cond_true.1		; <int> [#uses=2]

cond_true.1:		; preds = %entry
	unwind

invoke_cont.0:		; preds = %entry
	%tmp.4.i = cast int %tmp.7 to uint		; <uint> [#uses=0]
	%tmp.14.0.i5 = add int %tmp.7, -1		; <int> [#uses=1]
	br label %no_exit.i

no_exit.i:		; preds = %no_exit.i, %invoke_cont.0
	%tmp.14.0.i.0 = phi int [ %tmp.14.0.i, %no_exit.i ], [ %tmp.14.0.i5, %invoke_cont.0 ]		; <int> [#uses=1]
	%tmp.14.0.i = add int %tmp.14.0.i.0, -1		; <int> [#uses=1]
	br label %no_exit.i
}

declare int %_ZN5ArrayISt7complexIdEE8get_sizeERK10dim_vector()
