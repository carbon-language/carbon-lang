; RUN: llc < %s -march=x86 -mcpu=yonah -mattr=+sse2,-sse41 | FileCheck %s

; Without forcing instructions, fall back to the preferred PS domain.
; CHECK: vsel_double
; CHECK: xorps
; CHECK: andps
; CHECK: andnps
; CHECK: orps
; CHECK: ret

define void@vsel_double(<4 x double>* %v1, <4 x double>* %v2) {
  %A = load <4 x double>* %v1
  %B = load <4 x double>* %v2
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x double> %A, <4 x double> %B
  store <4 x double > %vsel, <4 x double>* %v1
  ret void
}


