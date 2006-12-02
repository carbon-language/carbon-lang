; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32

long %test() { ret long undef }
