; RUN: llvm-as < %s -o /dev/null -f


%T = type opaque
%X = global %T* null
%Y = global int* null

%T = type int
