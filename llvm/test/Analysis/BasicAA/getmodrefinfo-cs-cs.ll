; RUN: opt < %s -aa-eval -print-all-alias-modref-info -disable-output |& FileCheck %s

; CHECK: Just Ref: call void @ro() <-> call void @f0()

declare void @f0()
declare void @ro() readonly

define void @test() {
  call void @f0()
  call void @ro()
  ret void
}
