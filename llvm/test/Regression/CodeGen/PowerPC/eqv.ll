; RUN: llvm-as < %s | llc -march=ppc32 | grep eqv | wc -l  | grep 2 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep andc | wc -l | grep 2 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep orc | wc -l  | grep 2

int %test1(int %X, int %Y) {
	%A = xor int %X, %Y
	%B = xor int %A, -1
	ret int %B
}

int %test2(int %X, int %Y) {
	%A = xor int %X, %Y
	%B = xor int %A, -1
	ret int %B
}

int %test3(int %X, int %Y) {
	%A = xor int %Y, -1
	%B = and int %X, %A
	ret int %B
}

int %test4(int %X, int %Y) {
	%A = xor int %Y, -1
	%B = or  int %X, %A
	ret int %B
}

int %test5(int %X, int %Y) {
	%A = xor int %X, -1
	%B = and int %A, %Y
	ret int %B
}

int %test6(int %X, int %Y) {
	%A = xor int %X, -1
	%B = or  int %A, %Y
	ret int %B
}
