; RUN: llc -march=x86-64 < %s | FileCheck %s

; Codegen should be able to use a 32-bit shift instead of a 64-bit shift.
; CHECK: shll $16

define fastcc void @test(i32 %level, i64 %a, i64 %b, i64 %c, i64 %d, i32* %p) nounwind {
if.end523:                                        ; preds = %if.end453
  %conv7981749 = zext i32 %level to i64           ; <i64> [#uses=1]
  %and799 = shl i64 %conv7981749, 16              ; <i64> [#uses=1]
  %shl800 = and i64 %and799, 16711680             ; <i64> [#uses=1]
  %or801 = or i64 %shl800, %a                     ; <i64> [#uses=1]
  %or806 = or i64 %or801, %b                      ; <i64> [#uses=1]
  %or811 = or i64 %or806, %c                      ; <i64> [#uses=1]
  %or819 = or i64 %or811, %d                      ; <i64> [#uses=1]
  %conv820 = trunc i64 %or819 to i32              ; <i32> [#uses=1]
  store i32 %conv820, i32* %p
  ret void
}
