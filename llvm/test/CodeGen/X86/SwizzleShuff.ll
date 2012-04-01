; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; Check that we perform a scalar XOR on i32.

; CHECK: pull_bitcast
; CHECK: xorl
; CHECK: ret
define void @pull_bitcast (<4 x i8>* %pA, <4 x i8>* %pB) {
  %A = load <4 x i8>* %pA
  %B = load <4 x i8>* %pB
  %C = xor <4 x i8> %A, %B
  store <4 x i8> %C, <4 x i8>* %pA
  ret void
}
