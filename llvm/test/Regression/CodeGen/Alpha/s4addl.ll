; Make sure this testcase codegens to the bic instruction
; RUN: llvm-as < %s | llc -march=alpha | grep 's4addl'

; ModuleID = 'test.o'
deplibs = [ "c", "crtend" ]

implementation   ; Functions:

int %foo(int %x, int %y) {
entry:
	%tmp.1 = shl int %y, ubyte 2		; <int> [#uses=1]
	%tmp.3 = add int %tmp.1, %x		; <int> [#uses=1]
	ret int %tmp.3
}
