; RUN: llvm-as %s -o %t.o

; RUN: ld -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o -o %t2.o
; RUN: llvm-dis %t2.o -o - | FileCheck %s

; RUN: ld -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=also-emit-llvm \
; RUN:    -shared %t.o -o %t3.o
; RUN: llvm-dis %t3.o.bc -o /dev/null

; RUN: ld -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=also-emit-llvm=%t4 \
; RUN:    -shared %t.o -o %t3.o
; RUN: llvm-dis %t4 -o /dev/null

target triple = "x86_64-unknown-linux-gnu"

; CHECK: define internal void @f()
define hidden void @f() {
  ret void
}

; CHECK: define hidden void @g()
define hidden void @g() {
  ret void
}

@llvm.used = appending global [1 x i8*] [ i8* bitcast (void ()* @g to i8*)]
