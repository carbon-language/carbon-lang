;; Test that this FP immediate is stored in the constant pool as a float.

; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -mattr=-sse2,-sse3 | \
; RUN:   grep {.long.1123418112}

double %D() { ret double 123.0 }
