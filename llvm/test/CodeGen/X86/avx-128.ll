; RUN: llc < %s -mtriple=x86_64-apple-darwin -march=x86 -mcpu=corei7 -mattr=avx | FileCheck %s

@z = common global <4 x float> zeroinitializer, align 16

define void @zero() nounwind ssp {
entry:
  ; CHECK: vpxor
  ; CHECK: vmovaps
  store <4 x float> zeroinitializer, <4 x float>* @z, align 16
  ret void
}

