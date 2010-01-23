; RUN: opt -deadargelim -S %s | FileCheck %s

define void @test(i32) {
  ret void
}

define void @foo() {
  call void @test(i32 0)
  ret void
; CHECK: @foo
; CHECK: i32 undef
}
