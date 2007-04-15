; Make sure this testcase codegens to the lda -1 instruction
; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha | grep {\\-1}

implementation   ; Functions:

long %bar() {
entry:
	ret long -1
}
