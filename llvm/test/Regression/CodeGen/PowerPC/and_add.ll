; RUN: llvm-as < %s | llc -march=ppc32 | grep slwi &&
; RUN: llvm-as < %s | llc -march=ppc32 | not grep addi &&
; RUN: llvm-as < %s | llc -march=ppc32 | not grep rlwinm

int %test(int %A) {
  %B = mul int %A, 8  ;; shift
  %C = add int %B, 7  ;; dead, no demanded bits.
  %D = and int %C, -8 ;; dead once add is gone.
  ret int %D
}

