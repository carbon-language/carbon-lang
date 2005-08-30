; RUN: llvm-as < %s | llc -march=ppc32 | not grep 'ori\|lis'
; XFAIL: *

int %test(int %X) {
	%Y = and int %X, 32769   ; andi. r3, r3, 32769
	ret int %Y
}

int %test2(int %X) {
	%Y = and int %X, -2147418112 ; andis. r3, r3, 32769
	ret int %Y
}

