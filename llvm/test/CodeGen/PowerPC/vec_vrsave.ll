; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mcpu=g5 -o %t -f
; RUN: grep vrlw %t
; RUN: not grep spr %t
; RUN: not grep vrsave %t

<4 x int> %test_rol() {
        ret <4 x int> < int -11534337, int -11534337, int -11534337, int -11534337 >
}

<4 x int> %test_arg(<4 x int> %A, <4 x int> %B) {
	%C = add <4 x int> %A, %B
        ret <4 x int> %C
}

