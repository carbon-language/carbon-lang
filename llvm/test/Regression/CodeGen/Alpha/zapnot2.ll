; Make sure this testcase codegens to the zapnot instruction
; RUN: llvm-as < %s | llc -march=alpha | grep 'zapnot'

implementation   ; Functions:

long %bar(long %x) {
entry:
	%tmp.1 = and long %x, 16711935 		; <long> [#uses=1]
	ret long %tmp.1
}
