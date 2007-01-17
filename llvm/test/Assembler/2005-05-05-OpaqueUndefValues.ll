; RUN: llvm-upgrade < %s | llvm-as | llvm-dis | llvm-as

%t = type opaque
%x = global %t undef
