; RUN: opt -analyze -scalar-evolution %s -S | FileCheck %s

define i16 @test1(i8 %x) {
  %A = zext i8 %x to i12
  %B = sext i12 %A to i16
; CHECK: zext i8 %x to i16
  ret i16 %B
}

define i8 @test2(i8 %x) {
  %A = zext i8 %x to i16
  %B = add i16 %A, 1025
  %C = trunc i16 %B to i8
; CHECK: (1 + %x)
  ret i8 %C
}

define i8 @test3(i8 %x) {
  %A = zext i8 %x to i16
  %B = mul i16 %A, 1027
  %C = trunc i16 %B to i8
; CHECK: (3 * %x)
  ret i8 %C
}
