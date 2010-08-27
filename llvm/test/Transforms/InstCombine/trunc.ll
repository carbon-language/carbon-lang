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

define i32 @test5(i32 %A) {
  %B = zext i32 %A to i128
  %C = lshr i128 %B, 16
  %D = trunc i128 %C to i32
  ret i32 %D
; CHECK: @test5
; CHECK: %C = lshr i32 %A, 16
; CHECK: ret i32 %C
}

define i32 @test6(i64 %A) {
  %B = zext i64 %A to i128
  %C = lshr i128 %B, 32
  %D = trunc i128 %C to i32
  ret i32 %D
; CHECK: @test6
; CHECK: %C = lshr i64 %A, 32
; CHECK: %D = trunc i64 %C to i32
; CHECK: ret i32 %D
}

define i92 @test7(i64 %A) {
  %B = zext i64 %A to i128
  %C = lshr i128 %B, 32
  %D = trunc i128 %C to i92
  ret i92 %D
; CHECK: @test7
; CHECK: %C = lshr i64 %A, 32
; CHECK: %D = zext i64 %C to i92
; CHECK: ret i92 %D
}