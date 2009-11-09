; RUN: opt < %s -abcd -S | FileCheck %s

define void @test() {
; CHECK: @test
; CHECK-NOT: br i1 %tmp95
; CHECK: ret void
entry:
  br label %bb19

bb:
  br label %bb1

bb1:
  %tmp7 = icmp sgt i32 %tmp94, 1
  br i1 %tmp7, label %bb.i.i, label %return

bb.i.i:
  br label %return

bb19:
  %tmp94 = ashr i32 undef, 3
  %tmp95 = icmp sgt i32 %tmp94, 16
  br i1 %tmp95, label %bb, label %return

return:
  ret void
}
