; RUN: llvm-as < %s | opt -deadargelim | llvm-dis | not grep DEAD

implementation

internal int %test(int %DEADARG) {  ; Dead arg only used by dead retval
	ret int %DEADARG
}

int %test2(int %A) {
	%DEAD = call int %test(int %A)
	ret int 123
}

int %test3() {
	%X = call int %test2(int 3232)
	%Y = add int %X, -123
	ret int %Y
}

