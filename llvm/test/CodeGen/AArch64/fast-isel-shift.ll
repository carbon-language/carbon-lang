; RUN: llc -fast-isel -fast-isel-abort -mtriple=arm64-apple-darwin < %s | FileCheck %s

; CHECK-LABEL: lsl_i8
; CHECK: ubfiz {{w[0-9]*}}, {{w[0-9]*}}, #4, #4
define zeroext i8 @lsl_i8(i8 %a) {
  %1 = shl i8 %a, 4
  ret i8 %1
}

; CHECK-LABEL: lsl_i16
; CHECK: ubfiz {{w[0-9]*}}, {{w[0-9]*}}, #8, #8
define zeroext i16 @lsl_i16(i16 %a) {
  %1 = shl i16 %a, 8
  ret i16 %1
}

; CHECK-LABEL: lsl_i32
; CHECK: lsl {{w[0-9]*}}, {{w[0-9]*}}, #16
define zeroext i32 @lsl_i32(i32 %a) {
  %1 = shl i32 %a, 16
  ret i32 %1
}

; FIXME: This shouldn't use the variable shift version.
; CHECK-LABEL: lsl_i64
; CHECK: lsl {{x[0-9]*}}, {{x[0-9]*}}, {{x[0-9]*}}
define i64 @lsl_i64(i64 %a) {
  %1 = shl i64 %a, 32
  ret i64 %1
}

; CHECK-LABEL: lsr_i8
; CHECK: ubfx {{w[0-9]*}}, {{w[0-9]*}}, #4, #4
define zeroext i8 @lsr_i8(i8 %a) {
  %1 = lshr i8 %a, 4
  ret i8 %1
}

; CHECK-LABEL: lsr_i16
; CHECK: ubfx {{w[0-9]*}}, {{w[0-9]*}}, #8, #8
define zeroext i16 @lsr_i16(i16 %a) {
  %1 = lshr i16 %a, 8
  ret i16 %1
}

; CHECK-LABEL: lsr_i32
; CHECK: lsr {{w[0-9]*}}, {{w[0-9]*}}, #16
define zeroext i32 @lsr_i32(i32 %a) {
  %1 = lshr i32 %a, 16
  ret i32 %1
}

; FIXME: This shouldn't use the variable shift version.
; CHECK-LABEL: lsr_i64
; CHECK: lsr {{x[0-9]*}}, {{x[0-9]*}}, {{x[0-9]*}}
define i64 @lsr_i64(i64 %a) {
  %1 = lshr i64 %a, 32
  ret i64 %1
}

; CHECK-LABEL: asr_i8
; CHECK: sbfx {{w[0-9]*}}, {{w[0-9]*}}, #4, #4
define zeroext i8 @asr_i8(i8 %a) {
  %1 = ashr i8 %a, 4
  ret i8 %1
}

; CHECK-LABEL: asr_i16
; CHECK: sbfx {{w[0-9]*}}, {{w[0-9]*}}, #8, #8
define zeroext i16 @asr_i16(i16 %a) {
  %1 = ashr i16 %a, 8
  ret i16 %1
}

; CHECK-LABEL: asr_i32
; CHECK: asr {{w[0-9]*}}, {{w[0-9]*}}, #16
define zeroext i32 @asr_i32(i32 %a) {
  %1 = ashr i32 %a, 16
  ret i32 %1
}

; FIXME: This shouldn't use the variable shift version.
; CHECK-LABEL: asr_i64
; CHECK: asr {{x[0-9]*}}, {{x[0-9]*}}, {{x[0-9]*}}
define i64 @asr_i64(i64 %a) {
  %1 = ashr i64 %a, 32
  ret i64 %1
}

; CHECK-LABEL: shift_test1
; CHECK:       ubfiz {{w[0-9]*}}, {{w[0-9]*}}, #4, #4
; CHECK-NEXT:  sbfx  {{w[0-9]*}}, {{w[0-9]*}}, #4, #4
define i32 @shift_test1(i8 %a) {
  %1 = shl i8 %a, 4
  %2 = ashr i8 %1, 4
  %3 = sext i8 %2 to i32
  ret i32 %3
}

