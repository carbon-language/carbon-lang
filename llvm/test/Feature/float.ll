; RUN: llvm-as %s -o - | llvm-dis > Output/t1.ll
; RUN: llvm-as Output/t1.ll -o - | llvm-dis > Output/t2.ll
; RUN: diff Output/t1.ll Output/t2.ll

%F1     = global float 0x4010000000000000
%D1     = global double 0x4010000000000000
