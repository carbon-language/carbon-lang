; Test v2f64 rounding.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare <2 x double> @llvm.rint.v2f64(<2 x double>)
declare <2 x double> @llvm.nearbyint.v2f64(<2 x double>)
declare <2 x double> @llvm.floor.v2f64(<2 x double>)
declare <2 x double> @llvm.ceil.v2f64(<2 x double>)
declare <2 x double> @llvm.trunc.v2f64(<2 x double>)
declare <2 x double> @llvm.round.v2f64(<2 x double>)

define <2 x double> @f1(<2 x double> %val) {
; CHECK-LABEL: f1:
; CHECK: vfidb %v24, %v24, 0, 0
; CHECK: br %r14
  %res = call <2 x double> @llvm.rint.v2f64(<2 x double> %val)
  ret <2 x double> %res
}

define <2 x double> @f2(<2 x double> %val) {
; CHECK-LABEL: f2:
; CHECK: vfidb %v24, %v24, 4, 0
; CHECK: br %r14
  %res = call <2 x double> @llvm.nearbyint.v2f64(<2 x double> %val)
  ret <2 x double> %res
}

define <2 x double> @f3(<2 x double> %val) {
; CHECK-LABEL: f3:
; CHECK: vfidb %v24, %v24, 4, 7
; CHECK: br %r14
  %res = call <2 x double> @llvm.floor.v2f64(<2 x double> %val)
  ret <2 x double> %res
}

define <2 x double> @f4(<2 x double> %val) {
; CHECK-LABEL: f4:
; CHECK: vfidb %v24, %v24, 4, 6
; CHECK: br %r14
  %res = call <2 x double> @llvm.ceil.v2f64(<2 x double> %val)
  ret <2 x double> %res
}

define <2 x double> @f5(<2 x double> %val) {
; CHECK-LABEL: f5:
; CHECK: vfidb %v24, %v24, 4, 5
; CHECK: br %r14
  %res = call <2 x double> @llvm.trunc.v2f64(<2 x double> %val)
  ret <2 x double> %res
}

define <2 x double> @f6(<2 x double> %val) {
; CHECK-LABEL: f6:
; CHECK: vfidb %v24, %v24, 4, 1
; CHECK: br %r14
  %res = call <2 x double> @llvm.round.v2f64(<2 x double> %val)
  ret <2 x double> %res
}
