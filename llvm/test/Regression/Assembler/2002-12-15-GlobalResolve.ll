; RUN: llvm-as < %s -o /dev/null -f



%X = external global %T* 
%X = external global int *

%T = type int


implementation

