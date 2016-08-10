; RUN: opt < %s -correlated-propagation -S | FileCheck %s

; CHECK-LABEL: @test0(
define void @test0(i32 %a) {
entry:
  %cmp = icmp slt i32 %a, 100
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %add = add nsw i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test1(
define void @test1(i32 %a) {
entry:
  %cmp = icmp ult i32 %a, 100
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %add = add nuw nsw i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test2(
define void @test2(i32 %a) {
entry:
  %cmp = icmp ult i32 %a, -1
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %add = add nuw i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test3(
define void @test3(i32 %a) {
entry:
  %cmp = icmp ule i32 %a, -1
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %add = add i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test4(
define void @test4(i32 %a) {
entry:
  %cmp = icmp slt i32 %a, 2147483647
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %add = add nsw i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test5(
define void @test5(i32 %a) {
entry:
  %cmp = icmp sle i32 %a, 2147483647
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %add = add i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}
