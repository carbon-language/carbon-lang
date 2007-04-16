; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -o %t -f
; RUN: grep slwi %t
; RUN: not grep addi %t
; RUN: not grep rlwinm %t

int %test(int %A) {
  %B = mul int %A, 8  ;; shift
  %C = add int %B, 7  ;; dead, no demanded bits.
  %D = and int %C, -8 ;; dead once add is gone.
  ret int %D
}

