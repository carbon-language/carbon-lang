; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep 'add r0, r0, r1, lsl r2' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep 'bic r0, r0, r1, asr r2'

int %test1(int %X, int %Y, ubyte %sh) {
  %A = shl int %Y, ubyte %sh
  %B = add int %X, %A
  ret int %B
}

int %test2(int %X, int %Y, ubyte %sh) {
  %A = shr int %Y, ubyte %sh
  %B = xor int %A, -1
  %C = and int %X, %B
  ret int %C
}
