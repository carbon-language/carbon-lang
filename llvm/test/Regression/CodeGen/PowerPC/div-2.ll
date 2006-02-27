; RUN: llvm-as < %s | llc -march=ppc32 | not grep srawi &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep blr

int %test1(int %X) {
	%Y = and int %X, 15
	%Z = div int %Y, 4
	ret int %Z
}

int %test2(int %W) {
	%X = and int %W, 15
	%Y = sub int 16, %X
	%Z = div int %Y, 4
	ret int %Z
}

int %test3(int %W) {
	%X = and int %W, 15
	%Y = sub int 15, %X
	%Z = div int %Y, 4
	ret int %Z
}

int %test4(int %W) {
	%X = and int %W, 2
	%Y = sub int 5, %X
	%Z = div int %Y, 2
	ret int %Z
}
