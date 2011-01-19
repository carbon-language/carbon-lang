; RUN: opt -analyze -scalar-evolution %s -S | FileCheck %s

define i16 @test(i8 %x) {
  %A = zext i8 %x to i12
  %B = sext i12 %A to i16
; CHECK: zext i8 %x to i16
  ret i16 %B
}
