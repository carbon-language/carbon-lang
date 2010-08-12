; RUN: llc < %s -mtriple=x86_64-apple-darwin -march=x86 -mcpu=corei7 -mattr=avx | FileCheck %s

@x = common global <8 x float> zeroinitializer, align 32
@y = common global <4 x double> zeroinitializer, align 32

define void @zero() nounwind ssp {
entry:
  ; CHECK: vxorps
  ; CHECK: vmovaps
  ; CHECK: vmovaps
  store <8 x float> zeroinitializer, <8 x float>* @x, align 32
  store <4 x double> zeroinitializer, <4 x double>* @y, align 32
  ret void
}

