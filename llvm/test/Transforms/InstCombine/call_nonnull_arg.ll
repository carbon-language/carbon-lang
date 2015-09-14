; RUN: opt < %s -instcombine -S | FileCheck %s

; InstCombine should mark null-checked argument as nonnull at callsite
declare void @dummy(i32*, i32)

define void @test(i32* %a, i32 %b) {
; CHECK-LABEL: @test
; CHECK: call void @dummy(i32* nonnull %a, i32 %b)
entry:
  %cond1 = icmp eq i32* %a, null
  br i1 %cond1, label %dead, label %not_null
not_null:
  %cond2 = icmp eq i32 %b, 0
  br i1 %cond2, label %dead, label %not_zero
not_zero:
  call void @dummy(i32* %a, i32 %b)
  ret void
dead:
  unreachable
}
