; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @ripple(
; CHECK: add nsw i16 %tmp1, 1
define i32 @ripple(i16 signext %x) {
bb:
  %tmp = sext i16 %x to i32
  %tmp1 = and i32 %tmp, -5
  %tmp2 = trunc i32 %tmp1 to i16
  %tmp3 = sext i16 %tmp2 to i32
  %tmp4 = add i32 %tmp3, 1
  ret i32 %tmp4
}

; CHECK-LABEL: @ripplenot(
; CHECK: add i32 %tmp3, 4
define i32 @ripplenot(i16 signext %x) {
bb:
  %tmp = sext i16 %x to i32
  %tmp1 = and i32 %tmp, -3
  %tmp2 = trunc i32 %tmp1 to i16
  %tmp3 = sext i16 %tmp2 to i32
  %tmp4 = add i32 %tmp3, 4
  ret i32 %tmp4
}

; CHECK-LABEL: @oppositesign(
; CHECK: add nsw i16 %tmp1, 4
define i32 @oppositesign(i16 signext %x) {
bb:
  %tmp = sext i16 %x to i32
  %tmp1 = or i32 %tmp, 32768
  %tmp2 = trunc i32 %tmp1 to i16
  %tmp3 = sext i16 %tmp2 to i32
  %tmp4 = add i32 %tmp3, 4
  ret i32 %tmp4
}

; CHECK-LABEL: @ripplenot_var(
; CHECK: add i32 %tmp6, %tmp7
define i32 @ripplenot_var(i16 signext %x, i16 signext %y) {
bb:
  %tmp = sext i16 %x to i32
  %tmp1 = and i32 %tmp, -5
  %tmp2 = trunc i32 %tmp1 to i16
  %tmp3 = sext i16 %y to i32
  %tmp4 = or i32 %tmp3, 2
  %tmp5 = trunc i32 %tmp4 to i16
  %tmp6 = sext i16 %tmp5 to i32
  %tmp7 = sext i16 %tmp2 to i32
  %tmp8 = add i32 %tmp6, %tmp7
  ret i32 %tmp8
}
