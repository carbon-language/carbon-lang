; RUN: opt < %s -instcombine -S | FileCheck %s

; Instcombine should be able to eliminate all of these ext casts.

declare void @use(i32)

define i64 @test1(i64 %a) {
  %b = trunc i64 %a to i32
  %c = and i32 %b, 15
  %d = zext i32 %c to i64
  call void @use(i32 %b)
  ret i64 %d
; CHECK: @test1
; CHECK: %d = and i64 %a, 15
; CHECK: ret i64 %d
}
define i64 @test2(i64 %a) {
  %b = trunc i64 %a to i32
  %c = shl i32 %b, 4
  %q = ashr i32 %c, 4
  %d = sext i32 %q to i64
  call void @use(i32 %b)
  ret i64 %d
; CHECK: @test2
; CHECK: shl i64 %a, 36
; CHECK: %d = ashr i64 {{.*}}, 36
; CHECK: ret i64 %d
}
define i64 @test3(i64 %a) {
  %b = trunc i64 %a to i32
  %c = and i32 %b, 8
  %d = zext i32 %c to i64
  call void @use(i32 %b)
  ret i64 %d
; CHECK: @test3
; CHECK: %d = and i64 %a, 8
; CHECK: ret i64 %d
}
define i64 @test4(i64 %a) {
  %b = trunc i64 %a to i32
  %c = and i32 %b, 8
  %x = xor i32 %c, 8
  %d = zext i32 %x to i64
  call void @use(i32 %b)
  ret i64 %d
; CHECK: @test4
; CHECK: = and i64 %a, 8
; CHECK: %d = xor i64 {{.*}}, 8
; CHECK: ret i64 %d
}
