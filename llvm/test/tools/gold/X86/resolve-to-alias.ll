; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/resolve-to-alias.ll -o %t2.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o %t2.o -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s

define void @foo() {
  call void @bar()
  ret void
}
declare void @bar()

; CHECK: @bar = alias void (), void ()* @zed

; CHECK:      define void @foo() {
; CHECK-NEXT:   call void @bar()
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK:      define void @zed() {
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
