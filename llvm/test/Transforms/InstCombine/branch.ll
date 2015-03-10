; RUN: opt -instcombine -S < %s | FileCheck %s

define i32 @test(i32 %x) {
; CHECK-LABEL: @test
entry:
; CHECK-NOT: icmp
; CHECK: br i1 undef, 
  %cmp = icmp ult i32 %x, 7
  br i1 %cmp, label %merge, label %merge
merge:
; CHECK-LABEL: merge:
; CHECK: ret i32 %x
  ret i32 %x
}


