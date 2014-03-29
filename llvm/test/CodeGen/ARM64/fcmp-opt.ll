; RUN: llc < %s -march=arm64 -mcpu=cyclone | FileCheck %s
; rdar://10263824

define i1 @fcmp_float1(float %a) nounwind ssp {
entry:
; CHECK: @fcmp_float1
; CHECK: fcmp s0, #0.0
; CHECK: csinc w0, wzr, wzr, eq
  %cmp = fcmp une float %a, 0.000000e+00
  ret i1 %cmp
}

define i1 @fcmp_float2(float %a, float %b) nounwind ssp {
entry:
; CHECK: @fcmp_float2
; CHECK: fcmp s0, s1
; CHECK: csinc w0, wzr, wzr, eq
  %cmp = fcmp une float %a, %b
  ret i1 %cmp
}

define i1 @fcmp_double1(double %a) nounwind ssp {
entry:
; CHECK: @fcmp_double1
; CHECK: fcmp d0, #0.0
; CHECK: csinc w0, wzr, wzr, eq
  %cmp = fcmp une double %a, 0.000000e+00
  ret i1 %cmp
}

define i1 @fcmp_double2(double %a, double %b) nounwind ssp {
entry:
; CHECK: @fcmp_double2
; CHECK: fcmp d0, d1
; CHECK: csinc w0, wzr, wzr, eq
  %cmp = fcmp une double %a, %b
  ret i1 %cmp
}

; Check each fcmp condition
define float @fcmp_oeq(float %a, float %b) nounwind ssp {
; CHECK: @fcmp_oeq
; CHECK: fcmp s0, s1
; CHECK: csinc w{{[0-9]+}}, wzr, wzr, ne
  %cmp = fcmp oeq float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ogt(float %a, float %b) nounwind ssp {
; CHECK: @fcmp_ogt
; CHECK: fcmp s0, s1
; CHECK: csinc w{{[0-9]+}}, wzr, wzr, le
  %cmp = fcmp ogt float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_oge(float %a, float %b) nounwind ssp {
; CHECK: @fcmp_oge
; CHECK: fcmp s0, s1
; CHECK: csinc w{{[0-9]+}}, wzr, wzr, lt
  %cmp = fcmp oge float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_olt(float %a, float %b) nounwind ssp {
; CHECK: @fcmp_olt
; CHECK: fcmp s0, s1
; CHECK: csinc w{{[0-9]+}}, wzr, wzr, pl
  %cmp = fcmp olt float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ole(float %a, float %b) nounwind ssp {
; CHECK: @fcmp_ole
; CHECK: fcmp s0, s1
; CHECK: csinc w{{[0-9]+}}, wzr, wzr, hi
  %cmp = fcmp ole float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ord(float %a, float %b) nounwind ssp {
; CHECK: @fcmp_ord
; CHECK: fcmp s0, s1
; CHECK: csinc w{{[0-9]+}}, wzr, wzr, vs
  %cmp = fcmp ord float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_uno(float %a, float %b) nounwind ssp {
; CHECK: @fcmp_uno
; CHECK: fcmp s0, s1
; CHECK: csinc w{{[0-9]+}}, wzr, wzr, vc
  %cmp = fcmp uno float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ugt(float %a, float %b) nounwind ssp {
; CHECK: @fcmp_ugt
; CHECK: fcmp s0, s1
; CHECK: csinc w{{[0-9]+}}, wzr, wzr, ls
  %cmp = fcmp ugt float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_uge(float %a, float %b) nounwind ssp {
; CHECK: @fcmp_uge
; CHECK: fcmp s0, s1
; CHECK: csinc w{{[0-9]+}}, wzr, wzr, mi
  %cmp = fcmp uge float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ult(float %a, float %b) nounwind ssp {
; CHECK: @fcmp_ult
; CHECK: fcmp s0, s1
; CHECK: csinc w{{[0-9]+}}, wzr, wzr, ge
  %cmp = fcmp ult float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ule(float %a, float %b) nounwind ssp {
; CHECK: @fcmp_ule
; CHECK: fcmp s0, s1
; CHECK: csinc w{{[0-9]+}}, wzr, wzr, gt
  %cmp = fcmp ule float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_une(float %a, float %b) nounwind ssp {
; CHECK: @fcmp_une
; CHECK: fcmp s0, s1
; CHECK: csinc w{{[0-9]+}}, wzr, wzr, eq
  %cmp = fcmp une float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

; Possible opportunity for improvement.  See comment in
; ARM64TargetLowering::LowerSETCC()
define float @fcmp_one(float %a, float %b) nounwind ssp {
; CHECK: @fcmp_one
;	fcmp	s0, s1
;	orr	w0, wzr, #0x1
;	csel	w1, w0, wzr, mi
;	csel	w0, w0, wzr, gt
  %cmp = fcmp one float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

; Possible opportunity for improvement.  See comment in
; ARM64TargetLowering::LowerSETCC()
define float @fcmp_ueq(float %a, float %b) nounwind ssp {
; CHECK: @fcmp_ueq
; CHECK: fcmp s0, s1
;        orr w0, wzr, #0x1
; CHECK: csel [[REG1:w[0-9]]], [[REG2:w[0-9]+]], wzr, eq
; CHECK: csel {{w[0-9]+}}, [[REG2]], [[REG1]], vs
  %cmp = fcmp ueq float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}
