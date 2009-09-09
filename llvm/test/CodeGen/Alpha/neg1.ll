; Make sure this testcase codegens to the lda -1 instruction
; RUN: llc < %s -march=alpha | grep {\\-1}

define i64 @bar() {
entry:
	ret i64 -1
}
