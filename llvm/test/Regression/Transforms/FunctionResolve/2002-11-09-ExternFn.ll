; RUN: if as < %s | opt -funcresolve > /dev/null
; RUN: then echo "opt ok"
; RUN: else exit 1   # Make sure opt doesn't abort!
; RUN: fi
;
; RUN: if as < %s | opt -funcresolve -instcombine | dis | grep '\.\.\.'
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

declare int %foo(...)
declare int %foo(int)

void %bar() {
	call int(...)* %foo(int 7)
	ret void
}
