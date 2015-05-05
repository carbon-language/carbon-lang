; Test vector negation.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a v16i8 negation.
define <16 x i8> @f1(<16 x i8> %dummy, <16 x i8> %val) {
; CHECK-LABEL: f1:
; CHECK: vlcb %v24, %v26
; CHECK: br %r14
  %ret = sub <16 x i8> zeroinitializer, %val
  ret <16 x i8> %ret
}

; Test a v8i16 negation.
define <8 x i16> @f2(<8 x i16> %dummy, <8 x i16> %val) {
; CHECK-LABEL: f2:
; CHECK: vlch %v24, %v26
; CHECK: br %r14
  %ret = sub <8 x i16> zeroinitializer, %val
  ret <8 x i16> %ret
}

; Test a v4i32 negation.
define <4 x i32> @f3(<4 x i32> %dummy, <4 x i32> %val) {
; CHECK-LABEL: f3:
; CHECK: vlcf %v24, %v26
; CHECK: br %r14
  %ret = sub <4 x i32> zeroinitializer, %val
  ret <4 x i32> %ret
}

; Test a v2i64 negation.
define <2 x i64> @f4(<2 x i64> %dummy, <2 x i64> %val) {
; CHECK-LABEL: f4:
; CHECK: vlcg %v24, %v26
; CHECK: br %r14
  %ret = sub <2 x i64> zeroinitializer, %val
  ret <2 x i64> %ret
}
