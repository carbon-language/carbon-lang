; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

; PR12540: ARM backend lowering of FP_ROUND v2f64 to v2f32.
define <2 x float> @vtrunc(<2 x double> %a) {
; CHECK: vcvt.f32.f64 [[S0:s[0-9]+]], [[D0:d[0-9]+]]
; CHECK: vcvt.f32.f64 [[S1:s[0-9]+]], [[D1:d[0-9]+]]
  %vt = fptrunc <2 x double> %a to <2 x float>
  ret <2 x float> %vt
}
