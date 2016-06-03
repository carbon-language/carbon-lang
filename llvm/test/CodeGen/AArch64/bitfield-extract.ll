; RUN: llc -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s

; CHECK-LABEL: @test1
; CHECK: sbfx {{x[0-9]+}}, x0, #23, #9
define i64 @test1(i32 %a) {
  %tmp = ashr i32 %a, 23
  %ext = sext i32 %tmp to i64
  %res = add i64 %ext, 1
  ret i64 %res
}
