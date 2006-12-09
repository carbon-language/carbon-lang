; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86
%a = extern_weak global int
%b = global int* %a
