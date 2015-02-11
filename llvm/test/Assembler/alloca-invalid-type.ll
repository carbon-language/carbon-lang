; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: pointer to this type is invalid

define void @test() {
entry:
  alloca metadata !{null}
  ret void
}
