; RUN: llvm-as %s -o %t.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o -o %t2.o
; RUN: llvm-dis %t2.o -o - | FileCheck %s

define void @foo() {
  call void @bar()
  ret void
}
define available_externally void @bar() {
  ret void
}

; CHECK: define available_externally void @bar() {
