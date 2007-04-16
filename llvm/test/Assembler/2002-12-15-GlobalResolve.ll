; RUN: llvm-upgrade < %s 2>/dev/null | llvm-as -o /dev/null -f
; RUN: llvm-upgrade < %s |& grep {Renaming global variable 'X' to}

%X = external global uint *
%X = external global %T* 
%X = external global int *

%T = type int

implementation
