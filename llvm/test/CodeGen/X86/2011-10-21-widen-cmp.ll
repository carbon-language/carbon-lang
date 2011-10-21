; RUN: llc < %s -march=x86-64 -mcpu=corei7 | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; Check that a <4 x float> compare is generated and that we are
; not stuck in an endless loop.

; CHECK: cmp_2_floats
; CHECK: cmpordps
; CHECK: ret

define void @cmp_2_floats() {
entry:
  %0 = fcmp oeq <2 x float> undef, undef
  %1 = select <2 x i1> %0, <2 x float> undef, <2 x float> undef
  store <2 x float> %1, <2 x float>* undef
  ret void
}

; CHECK: cmp_2_doubles
; CHECK: cmpordpd
; CHECK: blendvpd
; CHECK: ret
define void @cmp_2_doubles() {
entry:
  %0 = fcmp oeq <2 x double> undef, undef
  %1 = select <2 x i1> %0, <2 x double> undef, <2 x double> undef
  store <2 x double> %1, <2 x double>* undef
  ret void
}
