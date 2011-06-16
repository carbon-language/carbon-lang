; RUN: llc < %s -mtriple=x86_64-apple-darwin -march=x86 -mcpu=corei7 -mattr=avx | FileCheck %s

@z = common global <4 x float> zeroinitializer, align 16

define void @zero() nounwind ssp {
entry:
  ; CHECK: vxorps
  ; CHECK: vmovaps
  store <4 x float> zeroinitializer, <4 x float>* @z, align 16
  ret void
}

define void @fpext() nounwind uwtable {
entry:
  %f = alloca float, align 4
  %d = alloca double, align 8
  %tmp = load float* %f, align 4
  ; CHECK: vcvtss2sd
  %conv = fpext float %tmp to double
  store double %conv, double* %d, align 8
  ret void
}
