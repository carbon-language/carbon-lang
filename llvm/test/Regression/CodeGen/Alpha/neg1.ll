; Make sure this testcase codegens to the lda -1 instruction
; RUN: llvm-as < %s | llc -march=alpha | grep '\-1'

implementation   ; Functions:

long %bar() {
entry:
	ret long -1
}
