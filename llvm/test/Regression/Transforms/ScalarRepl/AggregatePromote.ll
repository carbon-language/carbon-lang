; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | not grep alloca &&
; RUN: llvm-as < %s | opt -scalarrepl -disable-output

target endian = big
target pointersize = 32
target triple = "powerpc-apple-darwin8.0.0"

long %test1(long %X) {
	%A = alloca long
	store long %X, long* %A
	%B = cast long* %A to int*
	%C = cast int* %B to sbyte*
	store sbyte 0, sbyte* %C
	%Y = load long *%A
	ret long %Y
}


sbyte %test2(long %X) {
	%X_addr = alloca long		; <long*> [#uses=2]
	store long %X, long* %X_addr
	%tmp.0 = cast long* %X_addr to int*		; <int*> [#uses=1]
	%tmp.1 = getelementptr int* %tmp.0, int 1		; <int*> [#uses=1]
	%tmp.2 = cast int* %tmp.1 to sbyte*
	%tmp.3 = getelementptr sbyte* %tmp.2, int 3
	%tmp.2 = load sbyte* %tmp.3		; <int> [#uses=1]
	ret sbyte %tmp.2
}

short %crafty(long %X) {
        %a = alloca { long }
        %tmp.0 = getelementptr { long }* %a, int 0, uint 0            ; <long*> [#uses=1]
        store long %X, long* %tmp.0
        %tmp.3 = cast { long }* %a to [4 x short]*            ; <[4 x short]*> [#uses=2]
        %tmp.4 = getelementptr [4 x short]* %tmp.3, int 0, int 3                ; <short*> [#uses=1]
        %tmp.5 = load short* %tmp.4             ; <short> [#uses=1]
        %tmp.8 = getelementptr [4 x short]* %tmp.3, int 0, int 2                ; <short*> [#uses=1]
        %tmp.9 = load short* %tmp.8             ; <short> [#uses=1]
        %tmp.10 = or short %tmp.9, %tmp.5               ; <short> [#uses=1]
        ret short %tmp.10
}

short %crafty2(long %X) {
        %a = alloca long 
        store long %X, long* %a
        %tmp.3 = cast long* %a to [4 x short]*            ; <[4 x short]*> [#uses=2]
        %tmp.4 = getelementptr [4 x short]* %tmp.3, int 0, int 3                ; <short*> [#uses=1]
        %tmp.5 = load short* %tmp.4             ; <short> [#uses=1]
        %tmp.8 = getelementptr [4 x short]* %tmp.3, int 0, int 2                ; <short*> [#uses=1]
        %tmp.9 = load short* %tmp.8             ; <short> [#uses=1]
        %tmp.10 = or short %tmp.9, %tmp.5               ; <short> [#uses=1]
        ret short %tmp.10
}

