; RUN: llvm-as %s -o - | llvm-dis > Output/t1.ll
; RUN: llvm-as Output/t1.ll -o - | llvm-dis > Output/t2.ll
; RUN: diff Output/t1.ll Output/t2.ll


module asm "this is an inline asm block"
module asm "this is another inline asm block"

