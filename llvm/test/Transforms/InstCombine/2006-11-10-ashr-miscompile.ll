; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep lshr
; Verify this is not turned into -1.

int %test(ubyte %amt) {
  %B = lshr int -1, ubyte %amt
  ret int %B
}
