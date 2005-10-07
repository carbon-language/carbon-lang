; RUN: llvm-as < %s | llc -march=ppc32 | not grep srawi &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep blr

int %test(int %X) {
	%Y = and int %X, 15
	%Z = div int %Y, 4
	ret int %Z
}
