; Test vector division.  There is no native support for this, so it's really
; a test of the operation legalization code.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a v16i8 division.
define <16 x i8> @f1(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f1:
; CHECK: vlvgp [[REG:%v[0-9]+]],
; CHECK-DAG: vlvgb [[REG]], {{%r[0-5]}}, 0
; CHECK-DAG: vlvgb [[REG]], {{%r[0-5]}}, 1
; CHECK-DAG: vlvgb [[REG]], {{%r[0-5]}}, 2
; CHECK-DAG: vlvgb [[REG]], {{%r[0-5]}}, 3
; CHECK-DAG: vlvgb [[REG]], {{%r[0-5]}}, 4
; CHECK-DAG: vlvgb [[REG]], {{%r[0-5]}}, 5
; CHECK-DAG: vlvgb [[REG]], {{%r[0-5]}}, 6
; CHECK-DAG: vlvgb [[REG]], {{%r[0-5]}}, 8
; CHECK-DAG: vlvgb [[REG]], {{%r[0-5]}}, 9
; CHECK-DAG: vlvgb [[REG]], {{%r[0-5]}}, 10
; CHECK-DAG: vlvgb [[REG]], {{%r[0-5]}}, 11
; CHECK-DAG: vlvgb [[REG]], {{%r[0-5]}}, 12
; CHECK-DAG: vlvgb [[REG]], {{%r[0-5]}}, 13
; CHECK-DAG: vlvgb [[REG]], {{%r[0-5]}}, 14
; CHECK: br %r14
  %ret = sdiv <16 x i8> %val1, %val2
  ret <16 x i8> %ret
}

; Test a v8i16 division.
define <8 x i16> @f2(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f2:
; CHECK: vlvgp [[REG:%v[0-9]+]],
; CHECK-DAG: vlvgh [[REG]], {{%r[0-5]}}, 0
; CHECK-DAG: vlvgh [[REG]], {{%r[0-5]}}, 1
; CHECK-DAG: vlvgh [[REG]], {{%r[0-5]}}, 2
; CHECK-DAG: vlvgh [[REG]], {{%r[0-5]}}, 4
; CHECK-DAG: vlvgh [[REG]], {{%r[0-5]}}, 5
; CHECK-DAG: vlvgh [[REG]], {{%r[0-5]}}, 6
; CHECK: br %r14
  %ret = sdiv <8 x i16> %val1, %val2
  ret <8 x i16> %ret
}

; Test a v4i32 division.
define <4 x i32> @f3(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f3:
; CHECK: vlvgp [[REG:%v[0-9]+]],
; CHECK-DAG: vlvgf [[REG]], {{%r[0-5]}}, 0
; CHECK-DAG: vlvgf [[REG]], {{%r[0-5]}}, 2
; CHECK: br %r14
  %ret = sdiv <4 x i32> %val1, %val2
  ret <4 x i32> %ret
}

; Test a v2i64 division.
define <2 x i64> @f4(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f4:
; CHECK: vlvgp %v24,
; CHECK: br %r14
  %ret = sdiv <2 x i64> %val1, %val2
  ret <2 x i64> %ret
}
