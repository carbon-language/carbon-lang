; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @assume_add(i32 %a, i32 %b) {
; CHECK-LABEL: @assume_add(
  %1 = add i32 %a, %b
  %last_two_digits = and i32 %1, 3
  %2 = icmp eq i32 %last_two_digits, 0
  call void @llvm.assume(i1 %2)
  %3 = add i32 %1, 3
; CHECK: %3 = or i32 %1, 3
  ret i32 %3
}

declare void @llvm.assume(i1)
