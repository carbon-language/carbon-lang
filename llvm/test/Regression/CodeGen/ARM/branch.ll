; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm | grep bne &&
; RUN: llvm-as < %s | llc -march=arm | grep bge &&
; RUN: llvm-as < %s | llc -march=arm | grep bcs

void %f1(int %a, int %b, int* %v) {
entry:
	%tmp = seteq int %a, %b		; <bool> [#uses=1]
	br bool %tmp, label %cond_true, label %return

cond_true:		; preds = %entry
	store int 0, int* %v
	ret void

return:		; preds = %entry
	ret void
}

void %f2(int %a, int %b, int* %v) {
entry:
	%tmp = setlt int %a, %b		; <bool> [#uses=1]
	br bool %tmp, label %cond_true, label %return

cond_true:		; preds = %entry
	store int 0, int* %v
	ret void

return:		; preds = %entry
	ret void
}

void %f3(uint %a, uint %b, int* %v) {
entry:
	%tmp = setlt uint %a, %b		; <bool> [#uses=1]
	br bool %tmp, label %cond_true, label %return

cond_true:		; preds = %entry
	store int 0, int* %v
	ret void

return:		; preds = %entry
	ret void
}
