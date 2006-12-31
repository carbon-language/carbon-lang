; RUN: llvm-upgrade < %s | llvm-as -f -o %t.bc
; RUN: lli %t.bc > /dev/null

; test return instructions

void %test1() { ret void }
sbyte %test2() { ret sbyte 1 }
ubyte %test3() { ret ubyte 1 }
short %test4() { ret short -1 }
ushort %test5() { ret ushort 65535 }
int  %main() { ret int 0 }
uint %test6() { ret uint 4 }
long %test7() { ret long 0 }
ulong %test8() { ret ulong 0 }
float %test9() { ret float 1.0 }
double %test10() { ret double 2.0 }
