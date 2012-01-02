; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7 -promote-elements -mattr=+sse41 | FileCheck %s

; CHECK: foo
define <4 x double> @foo(<4 x double> %x, <4 x double> %y) {
  ; CHECK: cmpnlepd
  ; CHECK: psllq
  ; CHECK-NEXT: blendvpd
  ; CHECK: psllq
  ; CHECK-NEXT: blendvpd
  ; CHECK: ret
  %min_is_x = fcmp ult <4 x double> %x, %y
  %min = select <4 x i1> %min_is_x, <4 x double> %x, <4 x double> %y
  ret <4 x double> %min
}

