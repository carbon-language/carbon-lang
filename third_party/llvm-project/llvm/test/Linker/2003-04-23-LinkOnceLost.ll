; RUN: echo " define linkonce void @foo() { ret void } " > %t.ll
; RUN: llvm-link %s %t.ll -S | FileCheck %s
; CHECK: linkonce{{.*}}foo

declare void @foo()

define void @use_foo() {
  call void @foo()
  ret void
}
