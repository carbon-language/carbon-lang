; RUN: llc -fast-isel -fast-isel-abort=1 -mtriple=aarch64-apple-darwin -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: asr_zext_i1_i16
; CHECK:       uxth {{w[0-9]*}}, wzr
define zeroext i16 @asr_zext_i1_i16(i1 %b) {
  %1 = zext i1 %b to i16
  %2 = ashr i16 %1, 1
  ret i16 %2
}

; CHECK-LABEL: asr_sext_i1_i16
; CHECK:       sbfx [[REG1:w[0-9]+]], {{w[0-9]*}}, #0, #1
; CHECK-NEXT:  sxth {{w[0-9]*}}, [[REG1]]
define signext i16 @asr_sext_i1_i16(i1 %b) {
  %1 = sext i1 %b to i16
  %2 = ashr i16 %1, 1
  ret i16 %2
}

; CHECK-LABEL: asr_zext_i1_i32
; CHECK:       mov {{w[0-9]*}}, wzr
define i32 @asr_zext_i1_i32(i1 %b) {
  %1 = zext i1 %b to i32
  %2 = ashr i32 %1, 1
  ret i32 %2
}

; CHECK-LABEL: asr_sext_i1_i32
; CHECK:       sbfx  {{w[0-9]*}}, {{w[0-9]*}}, #0, #1
define i32 @asr_sext_i1_i32(i1 %b) {
  %1 = sext i1 %b to i32
  %2 = ashr i32 %1, 1
  ret i32 %2
}

; CHECK-LABEL: asr_zext_i1_i64
; CHECK:       mov {{x[0-9]*}}, xzr
define i64 @asr_zext_i1_i64(i1 %b) {
  %1 = zext i1 %b to i64
  %2 = ashr i64 %1, 1
  ret i64 %2
}

; CHECK-LABEL: asr_sext_i1_i64
; CHECK:       sbfx {{x[0-9]*}}, {{x[0-9]*}}, #0, #1
define i64 @asr_sext_i1_i64(i1 %b) {
  %1 = sext i1 %b to i64
  %2 = ashr i64 %1, 1
  ret i64 %2
}

; CHECK-LABEL: lsr_zext_i1_i16
; CHECK:       uxth {{w[0-9]*}}, wzr
define zeroext i16 @lsr_zext_i1_i16(i1 %b) {
  %1 = zext i1 %b to i16
  %2 = lshr i16 %1, 1
  ret i16 %2
}

; CHECK-LABEL: lsr_sext_i1_i16
; CHECK:       sbfx [[REG1:w[0-9]+]], {{w[0-9]*}}, #0, #1
; CHECK-NEXT:  ubfx [[REG2:w[0-9]+]], [[REG1]], #1, #15
; CHECK-NEXT:  sxth {{w[0-9]*}}, [[REG2]]
define signext i16 @lsr_sext_i1_i16(i1 %b) {
  %1 = sext i1 %b to i16
  %2 = lshr i16 %1, 1
  ret i16 %2
}

; CHECK-LABEL: lsr_zext_i1_i32
; CHECK:       mov {{w[0-9]*}}, wzr
define i32 @lsr_zext_i1_i32(i1 %b) {
  %1 = zext i1 %b to i32
  %2 = lshr i32 %1, 1
  ret i32 %2
}

; CHECK-LABEL: lsr_sext_i1_i32
; CHECK:       sbfx [[REG1:w[0-9]+]], {{w[0-9]*}}, #0, #1
; CHECK-NEXT:  lsr {{w[0-9]*}}, [[REG1:w[0-9]+]], #1
define i32 @lsr_sext_i1_i32(i1 %b) {
  %1 = sext i1 %b to i32
  %2 = lshr i32 %1, 1
  ret i32 %2
}

; CHECK-LABEL: lsr_zext_i1_i64
; CHECK:       mov {{x[0-9]*}}, xzr
define i64 @lsr_zext_i1_i64(i1 %b) {
  %1 = zext i1 %b to i64
  %2 = lshr i64 %1, 1
  ret i64 %2
}

; CHECK-LABEL: lsl_zext_i1_i16
; CHECK:       ubfiz {{w[0-9]*}}, {{w[0-9]*}}, #4, #1
define zeroext i16 @lsl_zext_i1_i16(i1 %b) {
  %1 = zext i1 %b to i16
  %2 = shl i16 %1, 4
  ret i16 %2
}

; CHECK-LABEL: lsl_sext_i1_i16
; CHECK:       sbfiz {{w[0-9]*}}, {{w[0-9]*}}, #4, #1
define signext i16 @lsl_sext_i1_i16(i1 %b) {
  %1 = sext i1 %b to i16
  %2 = shl i16 %1, 4
  ret i16 %2
}

; CHECK-LABEL: lsl_zext_i1_i32
; CHECK:       ubfiz {{w[0-9]*}}, {{w[0-9]*}}, #4, #1
define i32 @lsl_zext_i1_i32(i1 %b) {
  %1 = zext i1 %b to i32
  %2 = shl i32 %1, 4
  ret i32 %2
}

; CHECK-LABEL: lsl_sext_i1_i32
; CHECK:       sbfiz {{w[0-9]*}}, {{w[0-9]*}}, #4, #1
define i32 @lsl_sext_i1_i32(i1 %b) {
  %1 = sext i1 %b to i32
  %2 = shl i32 %1, 4
  ret i32 %2
}

; CHECK-LABEL: lsl_zext_i1_i64
; CHECK:       ubfiz {{x[0-9]*}}, {{x[0-9]*}}, #4, #1
define i64 @lsl_zext_i1_i64(i1 %b) {
  %1 = zext i1 %b to i64
  %2 = shl i64 %1, 4
  ret i64 %2
}

; CHECK-LABEL: lsl_sext_i1_i64
; CHECK:       sbfiz {{x[0-9]*}}, {{x[0-9]*}}, #4, #1
define i64 @lsl_sext_i1_i64(i1 %b) {
  %1 = sext i1 %b to i64
  %2 = shl i64 %1, 4
  ret i64 %2
}

; CHECK-LABEL: lslv_i8
; CHECK:       and [[REG1:w[0-9]+]], w1, #0xff
; CHECK-NEXT:  lsl [[REG2:w[0-9]+]], w0, [[REG1]]
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG2]], #0xff
define zeroext i8 @lslv_i8(i8 %a, i8 %b) {
  %1 = shl i8 %a, %b
  ret i8 %1
}

; CHECK-LABEL: lsl_i8
; CHECK:       ubfiz {{w[0-9]*}}, {{w[0-9]*}}, #4, #4
define zeroext i8 @lsl_i8(i8 %a) {
  %1 = shl i8 %a, 4
  ret i8 %1
}

; CHECK-LABEL: lsl_zext_i8_i16
; CHECK:       ubfiz {{w[0-9]*}}, {{w[0-9]*}}, #4, #8
define zeroext i16 @lsl_zext_i8_i16(i8 %b) {
  %1 = zext i8 %b to i16
  %2 = shl i16 %1, 4
  ret i16 %2
}

; CHECK-LABEL: lsl_sext_i8_i16
; CHECK:       sbfiz {{w[0-9]*}}, {{w[0-9]*}}, #4, #8
define signext i16 @lsl_sext_i8_i16(i8 %b) {
  %1 = sext i8 %b to i16
  %2 = shl i16 %1, 4
  ret i16 %2
}

; CHECK-LABEL: lsl_zext_i8_i32
; CHECK:       ubfiz {{w[0-9]*}}, {{w[0-9]*}}, #4, #8
define i32 @lsl_zext_i8_i32(i8 %b) {
  %1 = zext i8 %b to i32
  %2 = shl i32 %1, 4
  ret i32 %2
}

; CHECK-LABEL: lsl_sext_i8_i32
; CHECK:       sbfiz {{w[0-9]*}}, {{w[0-9]*}}, #4, #8
define i32 @lsl_sext_i8_i32(i8 %b) {
  %1 = sext i8 %b to i32
  %2 = shl i32 %1, 4
  ret i32 %2
}

; CHECK-LABEL: lsl_zext_i8_i64
; CHECK:       ubfiz {{x[0-9]*}}, {{x[0-9]*}}, #4, #8
define i64 @lsl_zext_i8_i64(i8 %b) {
  %1 = zext i8 %b to i64
  %2 = shl i64 %1, 4
  ret i64 %2
}

; CHECK-LABEL: lsl_sext_i8_i64
; CHECK:       sbfiz {{x[0-9]*}}, {{x[0-9]*}}, #4, #8
define i64 @lsl_sext_i8_i64(i8 %b) {
  %1 = sext i8 %b to i64
  %2 = shl i64 %1, 4
  ret i64 %2
}

; CHECK-LABEL: lslv_i16
; CHECK:       and [[REG1:w[0-9]+]], w1, #0xffff
; CHECK-NEXT:  lsl [[REG2:w[0-9]+]], w0, [[REG1]]
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG2]], #0xffff
define zeroext i16 @lslv_i16(i16 %a, i16 %b) {
  %1 = shl i16 %a, %b
  ret i16 %1
}

; CHECK-LABEL: lsl_i16
; CHECK:       ubfiz {{w[0-9]*}}, {{w[0-9]*}}, #8, #8
define zeroext i16 @lsl_i16(i16 %a) {
  %1 = shl i16 %a, 8
  ret i16 %1
}

; CHECK-LABEL: lsl_zext_i16_i32
; CHECK:       ubfiz {{w[0-9]*}}, {{w[0-9]*}}, #8, #16
define i32 @lsl_zext_i16_i32(i16 %b) {
  %1 = zext i16 %b to i32
  %2 = shl i32 %1, 8
  ret i32 %2
}

; CHECK-LABEL: lsl_sext_i16_i32
; CHECK:       sbfiz {{w[0-9]*}}, {{w[0-9]*}}, #8, #16
define i32 @lsl_sext_i16_i32(i16 %b) {
  %1 = sext i16 %b to i32
  %2 = shl i32 %1, 8
  ret i32 %2
}

; CHECK-LABEL: lsl_zext_i16_i64
; CHECK:       ubfiz {{x[0-9]*}}, {{x[0-9]*}}, #8, #16
define i64 @lsl_zext_i16_i64(i16 %b) {
  %1 = zext i16 %b to i64
  %2 = shl i64 %1, 8
  ret i64 %2
}

; CHECK-LABEL: lsl_sext_i16_i64
; CHECK:       sbfiz {{x[0-9]*}}, {{x[0-9]*}}, #8, #16
define i64 @lsl_sext_i16_i64(i16 %b) {
  %1 = sext i16 %b to i64
  %2 = shl i64 %1, 8
  ret i64 %2
}

; CHECK-LABEL: lslv_i32
; CHECK:       lsl {{w[0-9]*}}, w0, w1
define zeroext i32 @lslv_i32(i32 %a, i32 %b) {
  %1 = shl i32 %a, %b
  ret i32 %1
}

; CHECK-LABEL: lsl_i32
; CHECK:       lsl {{w[0-9]*}}, {{w[0-9]*}}, #16
define zeroext i32 @lsl_i32(i32 %a) {
  %1 = shl i32 %a, 16
  ret i32 %1
}

; CHECK-LABEL: lsl_zext_i32_i64
; CHECK:       ubfiz {{x[0-9]+}}, {{x[0-9]+}}, #16, #32
define i64 @lsl_zext_i32_i64(i32 %b) {
  %1 = zext i32 %b to i64
  %2 = shl i64 %1, 16
  ret i64 %2
}

; CHECK-LABEL: lsl_sext_i32_i64
; CHECK:       sbfiz {{x[0-9]+}}, {{x[0-9]+}}, #16, #32
define i64 @lsl_sext_i32_i64(i32 %b) {
  %1 = sext i32 %b to i64
  %2 = shl i64 %1, 16
  ret i64 %2
}

; CHECK-LABEL: lslv_i64
; CHECK:       lsl {{x[0-9]*}}, x0, x1
define i64 @lslv_i64(i64 %a, i64 %b) {
  %1 = shl i64 %a, %b
  ret i64 %1
}

; CHECK-LABEL: lsl_i64
; CHECK:       lsl {{x[0-9]*}}, {{x[0-9]*}}, #32
define i64 @lsl_i64(i64 %a) {
  %1 = shl i64 %a, 32
  ret i64 %1
}

; CHECK-LABEL: lsrv_i8
; CHECK:       and [[REG1:w[0-9]+]], w0, #0xff
; CHECK-NEXT:  and [[REG2:w[0-9]+]], w1, #0xff
; CHECK-NEXT:  lsr [[REG3:w[0-9]+]], [[REG1]], [[REG2]]
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG3]], #0xff
define zeroext i8 @lsrv_i8(i8 %a, i8 %b) {
  %1 = lshr i8 %a, %b
  ret i8 %1
}

; CHECK-LABEL: lsr_i8
; CHECK:       ubfx {{w[0-9]*}}, {{w[0-9]*}}, #4, #4
define zeroext i8 @lsr_i8(i8 %a) {
  %1 = lshr i8 %a, 4
  ret i8 %1
}

; CHECK-LABEL: lsr_zext_i8_i16
; CHECK:       ubfx {{w[0-9]*}}, {{w[0-9]*}}, #4, #4
define zeroext i16 @lsr_zext_i8_i16(i8 %b) {
  %1 = zext i8 %b to i16
  %2 = lshr i16 %1, 4
  ret i16 %2
}

; CHECK-LABEL: lsr_sext_i8_i16
; CHECK:       sxtb [[REG:w[0-9]+]], w0
; CHECK-NEXT:  ubfx {{w[0-9]*}}, [[REG]], #4, #12
define signext i16 @lsr_sext_i8_i16(i8 %b) {
  %1 = sext i8 %b to i16
  %2 = lshr i16 %1, 4
  ret i16 %2
}

; CHECK-LABEL: lsr_zext_i8_i32
; CHECK:       ubfx {{w[0-9]*}}, {{w[0-9]*}}, #4, #4
define i32 @lsr_zext_i8_i32(i8 %b) {
  %1 = zext i8 %b to i32
  %2 = lshr i32 %1, 4
  ret i32 %2
}

; CHECK-LABEL: lsr_sext_i8_i32
; CHECK:       sxtb [[REG:w[0-9]+]], w0
; CHECK-NEXT:  lsr {{w[0-9]*}}, [[REG]], #4
define i32 @lsr_sext_i8_i32(i8 %b) {
  %1 = sext i8 %b to i32
  %2 = lshr i32 %1, 4
  ret i32 %2
}

; CHECK-LABEL: lsrv_i16
; CHECK:       and [[REG1:w[0-9]+]], w0, #0xffff
; CHECK-NEXT:  and [[REG2:w[0-9]+]], w1, #0xffff
; CHECK-NEXT:  lsr [[REG3:w[0-9]+]], [[REG1]], [[REG2]]
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG3]], #0xffff
define zeroext i16 @lsrv_i16(i16 %a, i16 %b) {
  %1 = lshr i16 %a, %b
  ret i16 %1
}

; CHECK-LABEL: lsr_i16
; CHECK:       ubfx {{w[0-9]*}}, {{w[0-9]*}}, #8, #8
define zeroext i16 @lsr_i16(i16 %a) {
  %1 = lshr i16 %a, 8
  ret i16 %1
}

; CHECK-LABEL: lsrv_i32
; CHECK:       lsr {{w[0-9]*}}, w0, w1
define zeroext i32 @lsrv_i32(i32 %a, i32 %b) {
  %1 = lshr i32 %a, %b
  ret i32 %1
}

; CHECK-LABEL: lsr_i32
; CHECK:       lsr {{w[0-9]*}}, {{w[0-9]*}}, #16
define zeroext i32 @lsr_i32(i32 %a) {
  %1 = lshr i32 %a, 16
  ret i32 %1
}

; CHECK-LABEL: lsrv_i64
; CHECK:       lsr {{x[0-9]*}}, x0, x1
define i64 @lsrv_i64(i64 %a, i64 %b) {
  %1 = lshr i64 %a, %b
  ret i64 %1
}

; CHECK-LABEL: lsr_i64
; CHECK:       lsr {{x[0-9]*}}, {{x[0-9]*}}, #32
define i64 @lsr_i64(i64 %a) {
  %1 = lshr i64 %a, 32
  ret i64 %1
}

; CHECK-LABEL: asrv_i8
; CHECK:       sxtb [[REG1:w[0-9]+]], w0
; CHECK-NEXT:  and  [[REG2:w[0-9]+]], w1, #0xff
; CHECK-NEXT:  asr  [[REG3:w[0-9]+]], [[REG1]], [[REG2]]
; CHECK-NEXT:  and  {{w[0-9]+}}, [[REG3]], #0xff
define zeroext i8 @asrv_i8(i8 %a, i8 %b) {
  %1 = ashr i8 %a, %b
  ret i8 %1
}

; CHECK-LABEL: asr_i8
; CHECK:       sbfx {{w[0-9]*}}, {{w[0-9]*}}, #4, #4
define zeroext i8 @asr_i8(i8 %a) {
  %1 = ashr i8 %a, 4
  ret i8 %1
}

; CHECK-LABEL: asr_zext_i8_i16
; CHECK:       ubfx {{w[0-9]*}}, {{w[0-9]*}}, #4, #4
define zeroext i16 @asr_zext_i8_i16(i8 %b) {
  %1 = zext i8 %b to i16
  %2 = ashr i16 %1, 4
  ret i16 %2
}

; CHECK-LABEL: asr_sext_i8_i16
; CHECK:       sbfx {{w[0-9]*}}, {{w[0-9]*}}, #4, #4
define signext i16 @asr_sext_i8_i16(i8 %b) {
  %1 = sext i8 %b to i16
  %2 = ashr i16 %1, 4
  ret i16 %2
}

; CHECK-LABEL: asr_zext_i8_i32
; CHECK:       ubfx {{w[0-9]*}}, {{w[0-9]*}}, #4, #4
define i32 @asr_zext_i8_i32(i8 %b) {
  %1 = zext i8 %b to i32
  %2 = ashr i32 %1, 4
  ret i32 %2
}

; CHECK-LABEL: asr_sext_i8_i32
; CHECK:       sbfx {{w[0-9]*}}, {{w[0-9]*}}, #4, #4
define i32 @asr_sext_i8_i32(i8 %b) {
  %1 = sext i8 %b to i32
  %2 = ashr i32 %1, 4
  ret i32 %2
}

; CHECK-LABEL: asrv_i16
; CHECK:       sxth [[REG1:w[0-9]+]], w0
; CHECK-NEXT:  and  [[REG2:w[0-9]+]], w1, #0xffff
; CHECK-NEXT:  asr  [[REG3:w[0-9]+]], [[REG1]], [[REG2]]
; CHECK-NEXT:  and  {{w[0-9]+}}, [[REG3]], #0xffff
define zeroext i16 @asrv_i16(i16 %a, i16 %b) {
  %1 = ashr i16 %a, %b
  ret i16 %1
}

; CHECK-LABEL: asr_i16
; CHECK:       sbfx {{w[0-9]*}}, {{w[0-9]*}}, #8, #8
define zeroext i16 @asr_i16(i16 %a) {
  %1 = ashr i16 %a, 8
  ret i16 %1
}

; CHECK-LABEL: asrv_i32
; CHECK:       asr {{w[0-9]*}}, w0, w1
define zeroext i32 @asrv_i32(i32 %a, i32 %b) {
  %1 = ashr i32 %a, %b
  ret i32 %1
}

; CHECK-LABEL: asr_i32
; CHECK:       asr {{w[0-9]*}}, {{w[0-9]*}}, #16
define zeroext i32 @asr_i32(i32 %a) {
  %1 = ashr i32 %a, 16
  ret i32 %1
}

; CHECK-LABEL: asrv_i64
; CHECK:       asr {{x[0-9]*}}, x0, x1
define i64 @asrv_i64(i64 %a, i64 %b) {
  %1 = ashr i64 %a, %b
  ret i64 %1
}

; CHECK-LABEL: asr_i64
; CHECK:       asr {{x[0-9]*}}, {{x[0-9]*}}, #32
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

; Test zero shifts

; CHECK-LABEL: shl_zero
; CHECK-NOT:   lsl
define i32 @shl_zero(i32 %a) {
  %1 = shl i32 %a, 0
  ret i32 %1
}

; CHECK-LABEL: lshr_zero
; CHECK-NOT:   lsr
define i32 @lshr_zero(i32 %a) {
  %1 = lshr i32 %a, 0
  ret i32 %1
}

; CHECK-LABEL: ashr_zero
; CHECK-NOT:   asr
define i32 @ashr_zero(i32 %a) {
  %1 = ashr i32 %a, 0
  ret i32 %1
}

; CHECK-LABEL: shl_zext_zero
; CHECK:       ubfx x0, x0, #0, #32
define i64 @shl_zext_zero(i32 %a) {
  %1 = zext i32 %a to i64
  %2 = shl i64 %1, 0
  ret i64 %2
}

; CHECK-LABEL: lshr_zext_zero
; CHECK:       ubfx x0, x0, #0, #32
define i64 @lshr_zext_zero(i32 %a) {
  %1 = zext i32 %a to i64
  %2 = lshr i64 %1, 0
  ret i64 %2
}

; CHECK-LABEL: ashr_zext_zero
; CHECK:       ubfx x0, x0, #0, #32
define i64 @ashr_zext_zero(i32 %a) {
  %1 = zext i32 %a to i64
  %2 = ashr i64 %1, 0
  ret i64 %2
}

