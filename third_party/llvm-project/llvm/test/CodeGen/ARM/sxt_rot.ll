; RUN: llc -mtriple=arm-eabi -mattr=+v6 %s -o - | FileCheck %s --check-prefix=CHECK-V6
; RUN: llc -mtriple=arm-eabi -mattr=+v7 %s -o - | FileCheck %s --check-prefix=CHECK-V7

define i32 @test0(i8 %A) {
; CHECK-LABEL: test0
; CHECK-V6: sxtb r0, r0
; CHECK-V7: sxtb r0, r0
  %B = sext i8 %A to i32
  ret i32 %B
}

define signext i8 @test1(i32 %A) {
; CHECK-LABEL: test1
; CHECK-V6: lsr r0, r0, #8
; CHECK-V6: sxtb r0, r0
; CHECK-V6-NOT: sbfx
; CHECk-V7: sbfx r0, r0, #8, #8
  %B = lshr i32 %A, 8
  %C = shl i32 %A, 24
  %D = or i32 %B, %C
  %E = trunc i32 %D to i8
  ret i8 %E
}

define signext i32 @test2(i32 %A, i32 %X) {
; CHECK-LABEL: test2
; CHECK-V6: sxtab r0, r1, r0, ror #8
; CHECK-V7: sxtab r0, r1, r0, ror #8
  %B = lshr i32 %A, 8
  %C = shl i32 %A, 24
  %D = or i32 %B, %C
  %E = trunc i32 %D to i8
  %F = sext i8 %E to i32
  %G = add i32 %F, %X
  ret i32 %G
}

define signext i32 @test3(i32 %A, i32 %X) {
; CHECK-LABEL: test3
; CHECK-V6: sxtab r0, r1, r0, ror #16
; CHECK-V7: sxtab r0, r1, r0, ror #16
  %B = lshr i32 %A, 16
  %C = shl i32 %A, 16
  %D = or i32 %B, %C
  %E = trunc i32 %D to i8
  %F = sext i8 %E to i32
  %G = add i32 %F, %X
  ret i32 %G
}

define signext i32 @test4(i32 %A, i32 %X) {
; CHECK-LABEL: test4
; CHECK-V6: sxtah r0, r1, r0, ror #8
; CHECK-V7: sxtah r0, r1, r0, ror #8
  %B = lshr i32 %A, 8
  %C = shl i32 %A, 24
  %D = or i32 %B, %C
  %E = trunc i32 %D to i16
  %F = sext i16 %E to i32
  %G = add i32 %F, %X
  ret i32 %G
}

define signext i32 @test5(i32 %A, i32 %X) {
; CHECK-LABEL: test5
; CHECK-V6: sxtah r0, r1, r0, ror #24
; CHECK-V7: sxtah r0, r1, r0, ror #24
  %B = lshr i32 %A, 24
  %C = shl i32 %A, 8
  %D = or i32 %B, %C
  %E = trunc i32 %D to i16
  %F = sext i16 %E to i32
  %G = add i32 %F, %X
  ret i32 %G
}

define i32 @test6(i8 %A, i32 %X) {
; CHECK-LABEL: test6
; CHECK-V6: sxtab r0, r1, r0
; CHECK-V7: sxtab r0, r1, r0
  %sext = sext i8 %A to i32
  %add = add i32 %X, %sext
  ret i32 %add
}

define i32 @test7(i32 %A, i32 %X) {
; CHECK-LABEL: test7
; CHECK-V6: sxtab r0, r1, r0
; CHECK-V7: sxtab r0, r1, r0
  %shl = shl i32 %A, 24
  %shr = ashr i32 %shl, 24
  %add = add i32 %X, %shr
  ret i32 %add
}

define i32 @test8(i16 %A, i32 %X) {
; CHECK-LABEL: test8
; CHECK-V6: sxtah r0, r1, r0
; CHECK-V7: sxtah r0, r1, r0
  %sext = sext i16 %A to i32
  %add = add i32 %X, %sext
  ret i32 %add
}

define i32 @test9(i32 %A, i32 %X) {
; CHECK-LABEL: test9
; CHECK-V6: sxtah r0, r1, r0
; CHECK-V7: sxtah r0, r1, r0
  %shl = shl i32 %A, 16
  %shr = ashr i32 %shl, 16
  %add = add i32 %X, %shr
  ret i32 %add
}
