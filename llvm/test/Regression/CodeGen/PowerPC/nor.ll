; RUN: llvm-as < %s | llc -march=ppc32 | grep nor | wc -l | grep 2

int %test1(int %X) {
	%Y = xor int %X, -1
	ret int %Y
}

int %test2(int %X, int %Y) {
	%Z = or int %X, %Y
	%R = xor int %Z, -1
	ret int %R
}
