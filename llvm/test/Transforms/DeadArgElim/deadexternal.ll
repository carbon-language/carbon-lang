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

define void @f(i32 %X) {
entry:
  tail call void @sideeffect() nounwind
  ret void
}

declare void @sideeffect()

define void @g(i32 %n) {
entry:
  %add = add nsw i32 %n, 1
; CHECK: tail call void @f(i32 undef)
  tail call void @f(i32 %add)
  ret void
}

define void @h() {
entry:
  %i = alloca i32, align 4
  volatile store i32 10, i32* %i, align 4
; CHECK: %tmp = volatile load i32* %i, align 4
; CHECK-next: call void @f(i32 undef)
  %tmp = volatile load i32* %i, align 4
  call void @f(i32 %tmp)
  ret void
}
