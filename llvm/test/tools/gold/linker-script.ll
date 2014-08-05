; RUN: llvm-as %s -o %t.o

; RUN: ld -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o -o %t2.o \
; RUN:    -version-script=%p/Inputs/linker-script.export
; RUN: llvm-dis %t2.o -o - | FileCheck %s

; CHECK: define void @f()
define void @f() {
  ret void
}

; CHECK: define internal void @g()
define void @g() {
  ret void
}
