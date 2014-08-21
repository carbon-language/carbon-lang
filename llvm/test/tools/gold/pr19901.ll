; RUN: llc %s -o %t.o -filetype=obj -relocation-model=pic
; RUN: llvm-as %p/Inputs/pr19901-1.ll -o %t2.o
; RUN: ld -shared -o %t.so -plugin %llvmshlibdir/LLVMgold.so %t2.o %t.o
; RUN: llvm-objdump -d -symbolize %t.so | FileCheck %s

; CHECK: g:
; CHECK-NEXT: push
; CHECK-NEXT: callq f

target triple = "x86_64-unknown-linux-gnu"
define i32 @g() {
  call void @f()
  ret i32 0
}
define linkonce_odr hidden void @f() {
  ret void
}
