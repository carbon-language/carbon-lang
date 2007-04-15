; Make sure this testcase codegens to the ornot instruction
; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha | grep eqv

implementation   ; Functions:

long %bar(long %x) {
entry:
	%tmp.1 = xor long %x, -1  		; <long> [#uses=1]
	ret long %tmp.1
}
