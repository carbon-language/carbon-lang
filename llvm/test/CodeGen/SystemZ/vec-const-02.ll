; Test vector byte masks, v8i16 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test an all-zeros vector.
define <8 x i16> @f1() {
; CHECK-LABEL: f1:
; CHECK: vgbm %v24, 0
; CHECK: br %r14
  ret <8 x i16> zeroinitializer
}

; Test an all-ones vector.
define <8 x i16> @f2() {
; CHECK-LABEL: f2:
; CHECK: vgbm %v24, 65535
; CHECK: br %r14
  ret <8 x i16> <i16 -1, i16 -1, i16 -1, i16 -1,
                 i16 -1, i16 -1, i16 -1, i16 -1>
}

; Test a mixed vector (mask 0x8c76).
define <8 x i16> @f3() {
; CHECK-LABEL: f3:
; CHECK: vgbm %v24, 35958
; CHECK: br %r14
  ret <8 x i16> <i16 65280, i16 0, i16 65535, i16 0,
                 i16 255, i16 65535, i16 255, i16 65280>
}

; Test that undefs are treated as zero.
define <8 x i16> @f4() {
; CHECK-LABEL: f4:
; CHECK: vgbm %v24, 35958
; CHECK: br %r14
  ret <8 x i16> <i16 65280, i16 undef, i16 65535, i16 undef,
                 i16 255, i16 65535, i16 255, i16 65280>
}

; Test that we don't use VGBM if one of the bytes is not 0 or 0xff.
define <8 x i16> @f5() {
; CHECK-LABEL: f5:
; CHECK-NOT: vgbm
; CHECK: br %r14
  ret <8 x i16> <i16 65280, i16 0, i16 65535, i16 0,
                 i16 255, i16 65535, i16 256, i16 65280>
}
