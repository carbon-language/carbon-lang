; RUN: if as < %s | opt -funcresolve -instcombine | dis | grep '\.\.\.' | grep call
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

declare int %foo(...)

int %foo(int %x, float %y) {
	ret int %x
}

int %bar() {
	%x = call int(...)* %foo(double 12.5, int 48)
	ret int %x
}
