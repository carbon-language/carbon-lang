; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

define i64 @test_free_zext(i8* %a, i16* %b) {
; CHECK-LABEL: test_free_zext
; CHECK: ldrb w0, [x0]
; CHECK: ldrh w1, [x1]
; CHECK: add x0, x1, x0
  %1 = load i8* %a, align 1
  %conv = zext i8 %1 to i64
  %2 = load i16* %b, align 2
  %conv1 = zext i16 %2 to i64
  %add = add nsw i64 %conv1, %conv
  ret i64 %add
}
