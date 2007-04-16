; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | not grep .comm.*X,0

%X = linkonce global {} {}
