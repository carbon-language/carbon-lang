; RUN: llc < %s -march=arm64 -aarch64-neon-syntax=apple | FileCheck %s

define <2 x double> @f1(<2 x i32> %v) nounwind readnone {
; CHECK-LABEL: f1:
; CHECK: sshll.2d v0, v0, #0
; CHECK-NEXT: scvtf.2d v0, v0
; CHECK-NEXT: ret
  %conv = sitofp <2 x i32> %v to <2 x double>
  ret <2 x double> %conv
}
define <2 x double> @f2(<2 x i32> %v) nounwind readnone {
; CHECK-LABEL: f2:
; CHECK: ushll.2d v0, v0, #0
; CHECK-NEXT: ucvtf.2d v0, v0
; CHECK-NEXT: ret
  %conv = uitofp <2 x i32> %v to <2 x double>
  ret <2 x double> %conv
}

; CHECK: autogen_SD19655
; CHECK: scvtf
; CHECK: ret
define void @autogen_SD19655(<2 x i64>* %addr, <2 x float>* %addrfloat) {
  %T = load <2 x i64>* %addr
  %F = sitofp <2 x i64> %T to <2 x float>
  store <2 x float> %F, <2 x float>* %addrfloat
  ret void
}

