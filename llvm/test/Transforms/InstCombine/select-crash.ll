; RUN: opt < %s -instcombine -S | FileCheck %s
; Formerly crashed, PR8490.

define fastcc double @gimp_operation_color_balance_map(float %value, double %highlights) nounwind readnone inlinehint {
entry:
; CHECK: gimp_operation_color_balance_map
; CHECK: fsub double -0.000000
  %conv = fpext float %value to double
  %div = fdiv double %conv, 1.600000e+01
  %add = fadd double %div, 1.000000e+00
  %div1 = fdiv double 1.000000e+00, %add
  %sub = fsub double 1.075000e+00, %div1
  %sub24 = fsub double 1.000000e+00, %sub
  %add26 = fadd double %sub, 1.000000e+00
  %cmp86 = fcmp ogt double %highlights, 0.000000e+00
  %cond90 = select i1 %cmp86, double %sub24, double %add26
  %mul91 = fmul double %highlights, %cond90
  %add94 = fadd double undef, %mul91
  ret double %add94
}
