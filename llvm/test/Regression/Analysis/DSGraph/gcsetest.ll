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

int* %getp(%intpair* %P) {
	%pp = getelementptr %intpair* %P, long 0, ubyte 0
	%V = load int** %pp
	ret int *%V
}

int* %getq(%intpair* %P) {
	%pp = getelementptr %intpair* %P, long 0, ubyte 1
	%V = load int** %pp
	ret int *%V
}

int %test() {
	%C = call %intpair* %alloc_pair()
	%A = call int* %getp(%intpair* %C)
	%B = call int* %getp(%intpair* %C)
	%A1 = load int* %A

	store int 123, int* %B  ; Store does alias %A

	%A2 = load int* %A
	%x = sub int %A1, %A2
	ret int %x
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

int %test3() {
	%C = call %intpair* %alloc_pair()
	%P1 = call int* %getp(%intpair* %C)
	%P2 = call int* %getq(%intpair* %C)
	%X = load int* %P1
	store int 7, int* %P2
	%Y = load int* %P1
	%ELIM_x = sub int %X, %Y   ; Check field sensitivity
	ret int %ELIM_x
}
