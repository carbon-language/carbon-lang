; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel | grep movzx | wc -l | grep 1
; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel | grep movsx | wc -l | grep 1

%G1 = internal global ubyte 0		; <ubyte*> [#uses=1]
%G2 = internal global sbyte 0		; <sbyte*> [#uses=1]

implementation   ; Functions:

short %test1() {  ;; one zext
	%tmp.0 = load ubyte* %G1		; <ubyte> [#uses=1]
	%tmp.3 = cast ubyte %tmp.0 to short		; <short> [#uses=1]
	ret short %tmp.3
}

short %test2() {  ;; one sext
	%tmp.0 = load sbyte* %G2		; <sbyte> [#uses=1]
	%tmp.3 = cast sbyte %tmp.0 to short		; <short> [#uses=1]
	ret short %tmp.3
}
