; This shows where the function is called with the prototype indicating a
; return type doesn't exists, but it really does.
;
; RUN: if as < %s | opt -funcresolve > /dev/null
; RUN: then echo "opt ok"
; RUN: else exit 1   # Make sure opt doesn't abort!
; RUN: fi
;
; RUN: if as < %s | opt -funcresolve -instcombine | dis | grep '\.\.\.' | grep call
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

declare void %foo(...)

int %foo(int %x, float %y) {
	ret int %x
}

int %bar() {
	call void (...)* %foo(double 12.5, int 48)
	ret int 6
}
