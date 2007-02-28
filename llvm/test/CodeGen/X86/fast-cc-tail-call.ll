; XFAIL: *
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -x86-asm-syntax=intel | not grep call

fastcc int %bar(int %X, int(double, int) *%FP) {
     %Y = tail call fastcc int %FP(double 0.0, int %X)
     ret int %Y
}

