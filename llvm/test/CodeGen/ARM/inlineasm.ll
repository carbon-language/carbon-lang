; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+v6

uint %test1(uint %tmp54) {
  %tmp56 = tail call uint asm "uxtb16 $0,$1", "=r,r"( uint %tmp54 )
  ret uint %tmp56
}

void %test2() {
  %tmp1 = call long asm "ldmia $1!, {$0, ${0:H}}", "=r,==r,1"( int** null, int* null )
  %tmp1 = cast long %tmp1 to ulong
  %tmp2 = shr ulong %tmp1, ubyte 32
  %tmp3 = cast ulong %tmp2 to int
  %tmp4 = call int asm "pkhbt $0, $1, $2, lsl #16", "=r,r,r"( int 0, int %tmp3 )
  ret void
}
