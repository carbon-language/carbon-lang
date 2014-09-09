; RUN: llvm-as %s -o %t1.o
; RUN: llvm-as %p/Inputs/common.ll -o %t2.o

; RUN: ld -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t1.o %t2.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s

@a = common global i8 0, align 8

; CHECK: @a = common global i16 0, align 8
