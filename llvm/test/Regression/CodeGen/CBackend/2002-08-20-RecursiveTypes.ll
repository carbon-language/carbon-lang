; RUN: llvm-as < %s | llc -march=c

%MyIntList = uninitialized global { \2 *, int }

