; Instcombine was missing a test that caused it to make illegal transformations
; sometimes.  In this case, it transforms the sub into an add:
; RUN: echo foo
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep add
;


int "test"(int %i, int %j) {
	%A = mul int %i, %j
	%B = sub int 2, %A
	ret int %B
}
