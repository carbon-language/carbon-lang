; Test vector subtraction.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a v16i8 subtraction.
define <16 x i8> @f1(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f1:
; CHECK: vsb %v24, %v26, %v28
; CHECK: br %r14
  %ret = sub <16 x i8> %val1, %val2
  ret <16 x i8> %ret
}

; Test a v8i16 subtraction.
define <8 x i16> @f2(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f2:
; CHECK: vsh %v24, %v26, %v28
; CHECK: br %r14
  %ret = sub <8 x i16> %val1, %val2
  ret <8 x i16> %ret
}

; Test a v4i32 subtraction.
define <4 x i32> @f3(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f3:
; CHECK: vsf %v24, %v26, %v28
; CHECK: br %r14
  %ret = sub <4 x i32> %val1, %val2
  ret <4 x i32> %ret
}

; Test a v2i64 subtraction.
define <2 x i64> @f4(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f4:
; CHECK: vsg %v24, %v26, %v28
; CHECK: br %r14
  %ret = sub <2 x i64> %val1, %val2
  ret <2 x i64> %ret
}
