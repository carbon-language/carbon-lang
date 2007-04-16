; RUN: llvm-upgrade < %s | llvm-as | llc -march=sparc | \
; RUN:   grep xnor | wc -l | grep 2

int %test1(int %X, int %Y) {
	%A = xor int %X, %Y
	%B = xor int %A, -1
	ret int %B
}

int %test2(int %X, int %Y) {
	%A = xor int %X, -1
	%B = xor int %A, %Y
	ret int %B
}
