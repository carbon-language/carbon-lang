; This shows where the function is called with the prototype indicating a
; return type exists, but it really doesn't.
; RUN: if as < %s | opt -funcresolve > /dev/null
; RUN: then echo "opt ok"
; RUN: else exit 1   # Make sure opt doesn't abort!
; RUN: fi
;
; RUN: if as < %s | opt -funcresolve -instcombine | dis | grep '\.\.\.' | grep call
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

declare int %foo(...)

void %foo(int %x, float %y) {
	ret void
}

int %bar() {
	%x = call int(...)* %foo(double 12.5, int 48)
	ret int %x
}
