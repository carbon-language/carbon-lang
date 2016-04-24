; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/visibility.ll -o %t2.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    -shared %t.o %t2.o -o %t.so
; RUN: llvm-readobj -t %t.so | FileCheck %s

; CHECK:      Name: foo
; CHECK-NEXT: Value:
; CHECK-NEXT: Size: 1
; CHECK-NEXT: Binding: Global
; CHECK-NEXT: Type: Function
; CHECK-NEXT: Other [
; CHECK-NEXT:   STV_PROTECTED
; CHECK-NEXT: ]

define weak protected void @foo() {
  ret void
}
