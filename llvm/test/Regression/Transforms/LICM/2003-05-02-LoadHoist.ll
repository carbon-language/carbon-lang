; This testcase tests for a problem where LICM hoists loads out of a loop 
; despite the fact that calls to unknown functions may modify what is being 
; loaded from.  Basically if the load gets hoisted, the subtract gets turned
; into a constant zero.
;
; RUN: as < %s | opt -licm -load-vn -gcse -instcombine | dis | grep load

%X = global int 7
declare void %foo()

int %test(bool %c) {
	%A = load int *%X
	br label %Loop
Loop:
	call void %foo()
	%B = load int *%X  ;; Should not hoist this load!
	br bool %c, label %Loop, label %Out
Out:
	%C = sub int %A, %B
	ret int %C
}
