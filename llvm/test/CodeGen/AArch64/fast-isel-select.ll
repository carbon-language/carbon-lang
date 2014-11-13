; RUN: llc -mtriple=aarch64-apple-darwin                             -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort -verify-machineinstrs < %s | FileCheck %s

; First test the different supported value types for select.
define zeroext i1 @select_i1(i1 zeroext %c, i1 zeroext %a, i1 zeroext %b) {
; CHECK-LABEL: select_i1
; CHECK:       {{cmp w0, #0|tst w0, #0x1}}
; CHECK-NEXT:  csel {{w[0-9]+}}, w1, w2, ne
  %1 = select i1 %c, i1 %a, i1 %b
  ret i1 %1
}

define zeroext i8 @select_i8(i1 zeroext %c, i8 zeroext %a, i8 zeroext %b) {
; CHECK-LABEL: select_i8
; CHECK:       {{cmp w0, #0|tst w0, #0x1}}
; CHECK-NEXT:  csel {{w[0-9]+}}, w1, w2, ne
  %1 = select i1 %c, i8 %a, i8 %b
  ret i8 %1
}

define zeroext i16 @select_i16(i1 zeroext %c, i16 zeroext %a, i16 zeroext %b) {
; CHECK-LABEL: select_i16
; CHECK:       {{cmp w0, #0|tst w0, #0x1}}
; CHECK-NEXT:  csel {{w[0-9]+}}, w1, w2, ne
  %1 = select i1 %c, i16 %a, i16 %b
  ret i16 %1
}

define i32 @select_i32(i1 zeroext %c, i32 %a, i32 %b) {
; CHECK-LABEL: select_i32
; CHECK:       {{cmp w0, #0|tst w0, #0x1}}
; CHECK-NEXT:  csel {{w[0-9]+}}, w1, w2, ne
  %1 = select i1 %c, i32 %a, i32 %b
  ret i32 %1
}

define i64 @select_i64(i1 zeroext %c, i64 %a, i64 %b) {
; CHECK-LABEL: select_i64
; CHECK:       {{cmp w0, #0|tst w0, #0x1}}
; CHECK-NEXT:  csel {{x[0-9]+}}, x1, x2, ne
  %1 = select i1 %c, i64 %a, i64 %b
  ret i64 %1
}

define float @select_f32(i1 zeroext %c, float %a, float %b) {
; CHECK-LABEL: select_f32
; CHECK:       {{cmp w0, #0|tst w0, #0x1}}
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s0, s1, ne
  %1 = select i1 %c, float %a, float %b
  ret float %1
}

define double @select_f64(i1 zeroext %c, double %a, double %b) {
; CHECK-LABEL: select_f64
; CHECK:       {{cmp w0, #0|tst w0, #0x1}}
; CHECK-NEXT:  fcsel {{d[0-9]+}}, d0, d1, ne
  %1 = select i1 %c, double %a, double %b
  ret double %1
}
