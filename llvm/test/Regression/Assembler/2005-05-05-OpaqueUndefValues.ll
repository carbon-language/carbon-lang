; RUN: llvm-as < %s | llvm-dis | llvm-as

%t = type opaque
%x = global %t undef
