; RUN: llvm-upgrade < %s 2>/dev/null | llvm-as -o /dev/null -f &&
; RUN: llvm-upgrade < %s 2>&1 | grep "Cannot disambiguate global value '%X'" &&
; RUN: llvm-upgrade < %s 2>&1 | grep "Renaming global value '%X' to '%X.un"

%X = external global uint *
%X = external global %T* 
%X = external global int *

%T = type int

implementation
