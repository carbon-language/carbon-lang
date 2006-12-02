; RUN: llvm-upgrade < %s | llvm-as | opt -globaldce | llvm-dis | not grep global

%X = uninitialized global int
%Y = internal global int 7

