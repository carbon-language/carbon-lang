; RUN: llvm-as %s -o %t.o

; RUN: not ld -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o -o %t2.o 2>&1 | FileCheck %s

; CHECK: Unable to determine comdat of alias!

@g1 = global i32 1
@g2 = global i32 2

@a = alias inttoptr(i32 sub (i32 ptrtoint (i32* @g1 to i32),
                             i32 ptrtoint (i32* @g2 to i32)) to i32*)
