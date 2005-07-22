; Make sure this testcase codegens to the ctpop instruction
; RUN: llvm-as < %s | llc -march=alpha | grep -i 'subl $16,1,$0'

implementation   ; Functions:

int %foo(int %x) {
entry:
	%tmp.1 = add int %x, -1		; <int> [#uses=1]
	ret int %tmp.1
}
