; Test the handling of incoming vector arguments.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; This routine has 10 vector arguments, which fill up %v24-%v31 and
; the two double-wide stack slots at 160 and 176.
define <4 x i32> @foo(<4 x i32> %v1, <4 x i32> %v2, <4 x i32> %v3, <4 x i32> %v4,
                      <4 x i32> %v5, <4 x i32> %v6, <4 x i32> %v7, <4 x i32> %v8,
                      <4 x i32> %v9, <4 x i32> %v10) {
; CHECK-LABEL: foo:
; CHECK: vl [[REG1:%v[0-9]+]], 176(%r15)
; CHECK: vsf %v24, %v26, [[REG1]]
; CHECK: br %r14
  %y = sub <4 x i32> %v2, %v10
  ret <4 x i32> %y
}

; This routine has 10 vector arguments, which fill up %v24-%v31 and
; the two single-wide stack slots at 160 and 168.
define <4 x i8> @bar(<4 x i8> %v1, <4 x i8> %v2, <4 x i8> %v3, <4 x i8> %v4,
                     <4 x i8> %v5, <4 x i8> %v6, <4 x i8> %v7, <4 x i8> %v8,
                     <4 x i8> %v9, <4 x i8> %v10) {
; CHECK-LABEL: bar:
; CHECK: vlrepg [[REG1:%v[0-9]+]], 168(%r15)
; CHECK: vsb %v24, %v26, [[REG1]]
; CHECK: br %r14
  %y = sub <4 x i8> %v2, %v10
  ret <4 x i8> %y
}

