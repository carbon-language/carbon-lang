; RUN: llvm-as %s -o - | llvm-dis > Output/t1.ll
; RUN: llvm-as Output/t1.ll -o - | llvm-dis > Output/t2.ll
; RUN: diff Output/t1.ll Output/t2.ll


asm "this is an inline asm block"
asm "this is another inline asm block"

