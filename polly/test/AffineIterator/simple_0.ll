; RUN: opt %loadPolly %defaultOpts -print-scev-affine  -analyze  < %s | FileCheck %s

define i32 @f(i32 %a, i32 %b, i32 %c, i32 %d, i32* nocapture %x) nounwind readnone {
entry:
  %0 = shl i32 %a, 1                              ; <i32> [#uses=1]
; CHECK: 2 * %a + 0 * 1
  %1 = mul i32 %b, 3                              ; <i32> [#uses=1]
; CHECK: 3 * %b + 0 * 1
  %2 = shl i32 %d, 2                              ; <i32> [#uses=1]
; CHECK: 4 * %d + 0 * 1
  %3 = add nsw i32 %0, 5                          ; <i32> [#uses=1]
; CHECK: 2 * %a + 5 * 1
  %4 = add nsw i32 %3, %c                         ; <i32> [#uses=1]
; CHECK:  1 * %c + 2 * %a + 5 * 1
  %5 = add nsw i32 %4, %1                         ; <i32> [#uses=1]
; CHECK: 1 * %c + 3 * %b + 2 * %a + 5 * 1
  %6 = add nsw i32 %5, %2                         ; <i32> [#uses=1]
; CHECK: 1 * %c + 4 * %d + 3 * %b + 2 * %a + 5 * 1
  ret i32 %6
}
