; RUN: llvm-as < %s | llvm-dis &&
; RUN: llvm-as < %s | llvm-dis | grep 0x36A0000000000000

%A = global float bitcast (int 1 to float)
