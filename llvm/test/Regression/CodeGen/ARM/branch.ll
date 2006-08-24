; RUN: llvm-as < %s | llc -march=arm
void %f(int %a, int* %v) {
entry:
	%tmp = seteq int %a, 0		; <bool> [#uses=1]
	br bool %tmp, label %cond_true, label %return

cond_true:		; preds = %entry
	store int 0, int* %v
	ret void

return:		; preds = %entry
	ret void
}
