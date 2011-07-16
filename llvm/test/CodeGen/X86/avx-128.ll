; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

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

; CHECK: vcvtsi2sdq (%
define double @funcA(i64* nocapture %e) nounwind uwtable readonly ssp {
entry:
  %tmp1 = load i64* %e, align 8
  %conv = sitofp i64 %tmp1 to double
  ret double %conv
}

; CHECK: vcvtsi2sd (%
define double @funcB(i32* nocapture %e) nounwind uwtable readonly ssp {
entry:
  %tmp1 = load i32* %e, align 4
  %conv = sitofp i32 %tmp1 to double
  ret double %conv
}

; CHECK: vcvtsi2ss (%
define float @funcC(i32* nocapture %e) nounwind uwtable readonly ssp {
entry:
  %tmp1 = load i32* %e, align 4
  %conv = sitofp i32 %tmp1 to float
  ret float %conv
}

; CHECK: vcvtsi2ssq  (%
define float @funcD(i64* nocapture %e) nounwind uwtable readonly ssp {
entry:
  %tmp1 = load i64* %e, align 8
  %conv = sitofp i64 %tmp1 to float
  ret float %conv
}
