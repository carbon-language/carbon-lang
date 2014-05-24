; RUN: llc < %s -O0 -fast-isel-abort -verify-machineinstrs -mtriple=arm64-apple-darwin | FileCheck %s

define zeroext i1 @fcmp_float1(float %a) nounwind ssp {
entry:
; CHECK-LABEL: @fcmp_float1
; CHECK: fcmp s0, #0.0
; CHECK: cset w{{[0-9]+}}, ne
  %cmp = fcmp une float %a, 0.000000e+00
  ret i1 %cmp
}

define zeroext i1 @fcmp_float2(float %a, float %b) nounwind ssp {
entry:
; CHECK-LABEL: @fcmp_float2
; CHECK: fcmp s0, s1
; CHECK: cset w{{[0-9]+}}, ne
  %cmp = fcmp une float %a, %b
  ret i1 %cmp
}

define zeroext i1 @fcmp_double1(double %a) nounwind ssp {
entry:
; CHECK-LABEL: @fcmp_double1
; CHECK: fcmp d0, #0.0
; CHECK: cset w{{[0-9]+}}, ne
  %cmp = fcmp une double %a, 0.000000e+00
  ret i1 %cmp
}

define zeroext i1 @fcmp_double2(double %a, double %b) nounwind ssp {
entry:
; CHECK-LABEL: @fcmp_double2
; CHECK: fcmp d0, d1
; CHECK: cset w{{[0-9]+}}, ne
  %cmp = fcmp une double %a, %b
  ret i1 %cmp
}

; Check each fcmp condition
define float @fcmp_oeq(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_oeq
; CHECK: fcmp s0, s1
; CHECK: cset w{{[0-9]+}}, eq
  %cmp = fcmp oeq float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ogt(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_ogt
; CHECK: fcmp s0, s1
; CHECK: cset w{{[0-9]+}}, gt
  %cmp = fcmp ogt float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_oge(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_oge
; CHECK: fcmp s0, s1
; CHECK: cset w{{[0-9]+}}, ge
  %cmp = fcmp oge float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_olt(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_olt
; CHECK: fcmp s0, s1
; CHECK: cset w{{[0-9]+}}, mi
  %cmp = fcmp olt float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ole(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_ole
; CHECK: fcmp s0, s1
; CHECK: cset w{{[0-9]+}}, ls
  %cmp = fcmp ole float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ord(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_ord
; CHECK: fcmp s0, s1
; CHECK: cset {{w[0-9]+}}, vc
  %cmp = fcmp ord float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_uno(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_uno
; CHECK: fcmp s0, s1
; CHECK: cset {{w[0-9]+}}, vs
  %cmp = fcmp uno float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ugt(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_ugt
; CHECK: fcmp s0, s1
; CHECK: cset {{w[0-9]+}}, hi
  %cmp = fcmp ugt float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_uge(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_uge
; CHECK: fcmp s0, s1
; CHECK: cset {{w[0-9]+}}, pl
  %cmp = fcmp uge float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ult(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_ult
; CHECK: fcmp s0, s1
; CHECK: cset {{w[0-9]+}}, lt
  %cmp = fcmp ult float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ule(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_ule
; CHECK: fcmp s0, s1
; CHECK: cset {{w[0-9]+}}, le
  %cmp = fcmp ule float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_une(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_une
; CHECK: fcmp s0, s1
; CHECK: cset {{w[0-9]+}}, ne
  %cmp = fcmp une float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}
