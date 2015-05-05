; Test vector multiply-and-add.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a v16i8 multiply-and-add.
define <16 x i8> @f1(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2,
                     <16 x i8> %val3) {
; CHECK-LABEL: f1:
; CHECK: vmalb %v24, %v26, %v28, %v30
; CHECK: br %r14
  %mul = mul <16 x i8> %val1, %val2
  %ret = add <16 x i8> %mul, %val3
  ret <16 x i8> %ret
}

; Test a v8i16 multiply-and-add.
define <8 x i16> @f2(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2,
                     <8 x i16> %val3) {
; CHECK-LABEL: f2:
; CHECK: vmalhw %v24, %v26, %v28, %v30
; CHECK: br %r14
  %mul = mul <8 x i16> %val1, %val2
  %ret = add <8 x i16> %mul, %val3
  ret <8 x i16> %ret
}

; Test a v4i32 multiply-and-add.
define <4 x i32> @f3(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2,
                     <4 x i32> %val3) {
; CHECK-LABEL: f3:
; CHECK: vmalf %v24, %v26, %v28, %v30
; CHECK: br %r14
  %mul = mul <4 x i32> %val1, %val2
  %ret = add <4 x i32> %mul, %val3
  ret <4 x i32> %ret
}
