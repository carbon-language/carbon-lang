; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep vrlw &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | not grep spr &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | not grep vrsave

<4 x int> %test_rol() {
        ret <4 x int> < int -11534337, int -11534337, int -11534337, int -11534337 >
}

<4 x int> %test_arg(<4 x int> %A, <4 x int> %B) {
	%C = add <4 x int> %A, %B
        ret <4 x int> %C
}

