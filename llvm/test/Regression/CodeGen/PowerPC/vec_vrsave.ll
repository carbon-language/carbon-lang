; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep vrlw &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | not grep spr

<4 x int> %test_rol() {
        ret <4 x int> < int -11534337, int -11534337, int -11534337, int -11534337 >
}

