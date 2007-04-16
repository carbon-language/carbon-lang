; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm > %t 
; RUN: grep bne %t
; RUN: grep bge %t
; RUN: grep bhs %t
; RUN: grep blo %t

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

void %f4(uint %a, uint %b, int* %v) {
entry:
	%tmp = setlt uint %a, %b		; <bool> [#uses=1]
	br bool %tmp, label %return, label %cond_true

cond_true:		; preds = %entry
	store int 0, int* %v
	ret void

return:		; preds = %entry
	ret void
}
