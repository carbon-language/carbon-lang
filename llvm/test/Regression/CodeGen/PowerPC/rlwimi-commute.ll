; RUN: llvm-as < %s | llc -march=ppc32 | grep rlwimi &&
; RUN: llvm-as < %s | llc -march=ppc32 | not grep 'or '

; Make sure there is no register-register copies here.

void %test1(int *%A, int *%B, int *%D, int* %E) {
	%A = load int* %A
	%B = load int* %B
	%X = and int %A, 15
	%Y = and int %B, -16
	%Z = or int %X, %Y
	store int %Z, int* %D
	store int %A, int* %E
	ret void
}

void %test2(int *%A, int *%B, int *%D, int* %E) {
	%A = load int* %A
	%B = load int* %B
	%X = and int %A, 15
	%Y = and int %B, -16
	%Z = or int %X, %Y
	store int %Z, int* %D
	store int %B, int* %E
	ret void
}
