; Make sure that the constant propogator doesn't divide by zero!
;
; RUN: llvm-as < %s | opt -constprop
;

int "test"() {
	%R = div int 12, 0
	ret int %R
}

int "test2"() {
	%R = rem int 12, 0
	ret int %R
}
