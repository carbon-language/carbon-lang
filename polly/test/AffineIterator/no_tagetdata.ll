; RUN: opt %loadPolly %defaultOpts -print-scev-affine  -analyze  < %s | FileCheck %s

define i32 @f(i64 %a, i64 %b, i64 %c, [8 x i32]* nocapture %x) nounwind readonly {
entry:
  %0 = shl i64 %a, 1                              ; <i64> [#uses=1]
  %1 = add nsw i64 %0, %b                         ; <i64> [#uses=1]
; CHECK: 1 * %b + 2 * %a + 0 * 1
  %2 = shl i64 %1, 1                              ; <i64> [#uses=1]
; CHECK: 2 * %b + 4 * %a + 0 * 1
  %3 = add i64 %2, 2                              ; <i64> [#uses=1]
  %4 = mul i64 %a, 3                              ; <i64> [#uses=1]
  %5 = shl i64 %b, 2                              ; <i64> [#uses=1]
  %6 = add nsw i64 %4, 2                          ; <i64> [#uses=1]
  %7 = add nsw i64 %6, %c                         ; <i64> [#uses=1]
  %8 = add nsw i64 %7, %5                         ; <i64> [#uses=1]
  %9 = getelementptr inbounds [8 x i32]* %x, i64 %3, i64 %8 ; <i32*> [#uses=1]
; CHECK: 1 * %x + sizeof(i32) * %c + (35 * sizeof(i32)) * %a + (20 * sizeof(i32)) * %b + (18 * sizeof(i32)) * 1
  %10 = load i32* %9, align 4                     ; <i32> [#uses=1]
  ret i32 %10
}
