; Test that GCSE uses ds-aa to do alias analysis, which is capable of 
; disambiguating some cases.

; RUN: if as < %s | opt -ds-aa -load-vn -gcse -instcombine -dce | dis | grep ELIM
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

%intpair = type {int*, int*}
implementation

%intpair *%alloc_pair() {
	%Ap = malloc int
	%Bp = malloc int
 	%C  = malloc {int*, int*}
	%C1p = getelementptr {int*, int*}* %C, long 0, ubyte 0
	store int* %Ap, int** %C1p
	%C2p = getelementptr {int*, int*}* %C, long 0, ubyte 1
	store int* %Bp, int** %C2p
	ret %intpair* %C
}

int %test() {
	%C = call %intpair* %alloc_pair()
	%C1p = getelementptr %intpair* %C, long 0, ubyte 0
	%C2p = getelementptr %intpair* %C, long 0, ubyte 1
	%A = load int** %C1p
	%B = load int** %C2p
	%A1 = load int* %A

	store int 123, int* %B  ; Store cannot alias %A

	%A2 = load int* %A
	%ELIM_x = sub int %A1, %A2
	ret int %ELIM_x
}

int* %getp(%intpair* %P) {
	%pp = getelementptr %intpair* %P, long 0, ubyte 0
	%V = load int** %pp
	ret int *%V
}

int %test2() {   ; Test context sensitivity
	%C1 = call %intpair* %alloc_pair()
	%C2 = call %intpair* %alloc_pair()
	%P1 = call int* %getp(%intpair* %C1)
	%P2 = call int* %getp(%intpair* %C2)
	%X = load int* %P1
	store int 7, int* %P2
	%Y = load int* %P1
	%ELIM_x = sub int %X, %Y
	ret int %ELIM_x
}

