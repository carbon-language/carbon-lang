; RUN: llvm-as < %s | llc -march=ppc32

long %test() { ret long undef }
