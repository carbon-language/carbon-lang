; RUN: llvm-as < %s | llc -x86-asm-syntax=intel -enable-x86-fastcc | not grep call

fastcc int %bar(int %X, int(double, int) *%FP) {
     %Y = tail call fastcc int %FP(double 0.0, int %X)
     ret int %Y
}

