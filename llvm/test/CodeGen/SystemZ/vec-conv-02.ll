; Test conversions between different-sized float elements.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test cases where both elements of a v2f64 are converted to f32s.
define void @f1(<2 x double> %val, <2 x float> *%ptr) {
; CHECK-LABEL: f1:
; CHECK: vledb {{%v[0-9]+}}, %v24, 0, 0
; CHECK: br %r14
  %res = fptrunc <2 x double> %val to <2 x float>
  store <2 x float> %res, <2 x float> *%ptr
  ret void
}
