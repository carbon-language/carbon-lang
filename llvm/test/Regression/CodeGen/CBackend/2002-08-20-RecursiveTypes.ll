; RUN: llvm-upgrade < %s | llvm-as | llc -march=c

%MyIntList = uninitialized global { \2 *, int }

