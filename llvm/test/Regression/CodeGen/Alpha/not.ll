; Make sure this testcase codegens to the ornot instruction
; RUN: llvm-as < %s | llc -march=alpha | grep 'ornot'

implementation   ; Functions:

long %bar(long %x) {
entry:
	%tmp.1 = xor long %x, -1  		; <long> [#uses=1]
	ret long %tmp.1
}
