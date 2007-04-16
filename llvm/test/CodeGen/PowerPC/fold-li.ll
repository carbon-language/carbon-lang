; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | \
; RUN:   grep -v align | not grep li

;; Test that immediates are folded into these instructions correctly.

int %ADD(int %X) {
	%Y = add int %X, 65537
	ret int %Y
}

int %SUB(int %X) {
	%Y = sub int %X, 65537
	ret int %Y
}
