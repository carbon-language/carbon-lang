; RUN: llvm-as %s -o /dev/null -f
; XFAIL: *
int %f(int %a) {
entry:
	%tmp = seteq int %a, 4		; <bool> [#uses=1]
	br bool %tmp, label %cond_false, label %cond_true

cond_true:		; preds = %entry
	br label %return

cond_false:		; preds = %entry
	br label %return

return:		; preds = %cond_false, %cond_true
	%retval.0 = phi int [ 2, %cond_true ], [ 3, %cond_false ]		; <int> [#uses=1]
	ret int %retval.0
}
