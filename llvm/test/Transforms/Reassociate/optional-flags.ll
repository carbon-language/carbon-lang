; RUN: opt -S -reassociate < %s | FileCheck %s
; rdar://8944681

; Reassociate should clear optional flags like nsw when reassociating.

; CHECK: @test0
; CHECK: %y = add i64 %b, %a
; CHECK: %z = add i64 %y, %c
define i64 @test0(i64 %a, i64 %b, i64 %c) {
  %y = add nsw i64 %c, %b
  %z = add i64 %y, %a
  ret i64 %z
}

; CHECK: @test1
; CHECK: %y = add i64 %b, %a
; CHECK: %z = add i64 %y, %c
define i64 @test1(i64 %a, i64 %b, i64 %c) {
  %y = add i64 %c, %b
  %z = add nsw i64 %y, %a
  ret i64 %z
}
