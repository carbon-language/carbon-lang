; RUN: opt %loadPolly %defaultOpts -print-scev-affine  -analyze  < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @f(i32 %a, i32 %b, i32 %c, i64 %d, i8 signext %e, i32 %f, i32 %g, i32 %h) nounwind readnone {
entry:
  %0 = mul i32 %a, 3                              ; <i32> [#uses=1]
  %1 = mul i32 %b, 5                              ; <i32> [#uses=1]
  %2 = mul i32 %1, %c                             ; <i32> [#uses=1]
; CHECK: 5 * (%b * %c) + 0 * 1
  %3 = mul i32 %2, %f                             ; <i32> [#uses=1]
; CHECK: 5 * (%b * %c * %f) + 0 * 1
  %4 = sext i8 %e to i32                          ; <i32> [#uses=1]
  %5 = shl i32 %4, 2                              ; <i32> [#uses=1]
  %6 = trunc i64 %d to i32                        ; <i32> [#uses=1]
  %7 = mul i32 %6, %h                             ; <i32> [#uses=1]
  %8 = add nsw i32 %0, %g                         ; <i32> [#uses=1]
  %9 = add nsw i32 %8, %5                         ; <i32> [#uses=1]
  %10 = add nsw i32 %9, %3                        ; <i32> [#uses=1]
  %11 = add nsw i32 %10, %7                       ; <i32> [#uses=1]
; CHECK: 1 * %g + 1 * ((trunc i64 %d to i32) * %h) + 5 * (%b * %c * %f) + 4 * (sext i8 %e to i32) + 3 * %a + 0 * 1
  ret i32 %11
}
