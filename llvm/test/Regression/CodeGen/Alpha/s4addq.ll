; Make sure this testcase codegens to the S4ADDQ instruction
; RUN: llvm-as < %s | llc -march=alpha | grep 's4addq'

implementation   ; Functions:

long %bar(long %x, long %y) {
entry:
	%tmp.1 = shl long %x, ubyte 3		; <long> [#uses=1]
	%tmp.3 = add long %tmp.1, %y		; <long> [#uses=1]
	ret long %tmp.3
}
