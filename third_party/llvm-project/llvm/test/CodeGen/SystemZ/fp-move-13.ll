; Test f128 moves on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

; VR-to-VR moves.  Since f128s are passed by reference,
; we need to force a copy by other means.
define void @f1(fp128 *%x) {
; CHECK-LABEL: f1:
; CHECK: vlr
; CHECK: vleig
; CHECK: br %r14
  %val = load volatile fp128, fp128 *%x
  %t1 = bitcast fp128 %val to <2 x i64>
  %t2 = insertelement <2 x i64> %t1, i64 0, i32 0
  %res = bitcast <2 x i64> %t2 to fp128
  store volatile fp128 %res, fp128 *%x
  store volatile fp128 %val, fp128 *%x
  ret void
}

; Test 128-bit moves from GPRs to VRs.  i128 isn't a legitimate type,
; so this goes through memory.
define void @f2(fp128 *%a, i128 *%b) {
; CHECK-LABEL: f2:
; CHECK: vl
; CHECK: vst
; CHECK: br %r14
  %val = load i128, i128 *%b
  %res = bitcast i128 %val to fp128
  store fp128 %res, fp128 *%a
  ret void
}

; Test 128-bit moves from VRs to GPRs, with the same restriction as f2.
define void @f3(fp128 *%a, i128 *%b) {
; CHECK-LABEL: f3:
; CHECK: vl
; CHECK: vst
  %val = load fp128, fp128 *%a
  %res = bitcast fp128 %val to i128
  store i128 %res, i128 *%b
  ret void
}

