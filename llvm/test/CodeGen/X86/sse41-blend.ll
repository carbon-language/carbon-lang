; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7 -promote-elements -mattr=+sse41 | FileCheck %s

;CHECK: vsel_float
;CHECK: blendvps
;CHECK: ret
define <4 x float> @vsel_float(<4 x float> %v1, <4 x float> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x float> %v1, <4 x float> %v2
  ret <4 x float> %vsel
}


;CHECK: vsel_i32
;CHECK: blendvps
;CHECK: ret
define <4 x i32> @vsel_i32(<4 x i32> %v1, <4 x i32> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x i32> %v1, <4 x i32> %v2
  ret <4 x i32> %vsel
}


;CHECK: vsel_double
;CHECK: blendvpd
;CHECK: ret
define <4 x double> @vsel_double(<4 x double> %v1, <4 x double> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x double> %v1, <4 x double> %v2
  ret <4 x double> %vsel
}


;CHECK: vsel_i64
;CHECK: blendvpd
;CHECK: ret
define <4 x i64> @vsel_i64(<4 x i64> %v1, <4 x i64> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x i64> %v1, <4 x i64> %v2
  ret <4 x i64> %vsel
}


;CHECK: vsel_i8
;CHECK: pblendvb
;CHECK: ret
define <16 x i8> @vsel_i8(<16 x i8> %v1, <16 x i8> %v2) {
  %vsel = select <16 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <16 x i8> %v1, <16 x i8> %v2
  ret <16 x i8> %vsel
}

;; TEST blend + compares
; CHECK: A
define <2 x double> @A(<2 x double> %x, <2 x double> %y) {
  ; CHECK: cmplepd
  ; CHECK: blendvpd
  %max_is_x = fcmp oge <2 x double> %x, %y
  %max = select <2 x i1> %max_is_x, <2 x double> %x, <2 x double> %y
  ret <2 x double> %max
}

; CHECK: B
define <2 x double> @B(<2 x double> %x, <2 x double> %y) {
  ; CHECK: cmpltpd
  ; CHECK: blendvpd
  %max_is_x = fcmp ogt <2 x double> %x, %y
  %max = select <2 x i1> %max_is_x, <2 x double> %x, <2 x double> %y
  ret <2 x double> %max
}

