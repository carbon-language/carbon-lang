; RUN: llvm-upgrade < %s | llvm-as | llc -march=c


declare void %foo(...)


