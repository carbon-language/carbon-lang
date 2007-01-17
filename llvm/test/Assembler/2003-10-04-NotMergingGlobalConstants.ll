; RUN: llvm-upgrade < %s | llvm-as -o /dev/null -f

%T = type opaque
%X = global %T* null
%Y = global int* null

%T = type int
