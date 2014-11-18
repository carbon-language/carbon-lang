; RUN: opt -S -reassociate -dce < %s | FileCheck %s
; rdar://8944681

; Reassociate should clear optional flags like nsw when reassociating.

; CHECK-LABEL: @test0(
; CHECK: %z = add i64 %b, 2
define i64 @test0(i64 %a, i64 %b) {
  %x = add nsw i64 %a, 2
  %y = add nsw i64 %x, %b
  %z = sub nsw i64 %y, %a
  ret i64 %z
}

; CHECK-LABEL: @test1(
; CHECK: %y = mul i64 %a, 6
; CHECK: %z = sub nsw i64 %y, %a
define i64 @test1(i64 %a, i64 %b) {
  %x = add nsw i64 %a, %a
  %y = mul nsw i64 %x, 3
  %z = sub nsw i64 %y, %a
  ret i64 %z
}

; PR9215
; CHECK: %s = add nsw i32 %x, %y
define i32 @test2(i32 %x, i32 %y) {
  %s = add nsw i32 %x, %y
  ret i32 %s
}
