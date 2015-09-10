; RUN: opt < %s -instcombine -S | FileCheck %s

; InstCombine should mark null-checked argument as nonnull at callsite
declare void @dummy(i32*)

define void @test(i32* %a) {
; CHECK-LABEL: @test
; CHECK: call void @dummy(i32* nonnull %a)
entry:
  %cond = icmp eq i32* %a, null
  br i1 %cond, label %is_null, label %not_null
not_null:
  call void @dummy(i32* %a)
  ret void
is_null:
  unreachable
}
