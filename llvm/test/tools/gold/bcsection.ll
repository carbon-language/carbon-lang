; RUN: llvm-as -o %T/bcsection.bc %s

; RUN: llvm-mc -I=%T -filetype=obj -o %T/bcsection.bco %p/Inputs/bcsection.s
; RUN: llvm-nm -no-llvm-bc %T/bcsection.bco | count 0
; RUN: %gold -r -o %T/bcsection.o -plugin %llvmshlibdir/LLVMgold.so %T/bcsection.bco
; RUN: llvm-nm -no-llvm-bc %T/bcsection.o | FileCheck %s

; CHECK: main
define i32 @main() {
  ret i32 0
}
