; RUN: opt < %s -constprop -S | FileCheck %s

declare double @cos(double)

declare double @sin(double)

declare double @tan(double)

declare double @sqrt(double)

define double @T() {
; CHECK: @T
; CHECK-NOT: call
; CHECK: ret
  %A = call double @cos(double 0.000000e+00)
  %B = call double @sin(double 0.000000e+00)
  %a = fadd double %A, %B
  %C = call double @tan(double 0.000000e+00)
  %b = fadd double %a, %C
  %D = call double @sqrt(double 4.000000e+00)
  %c = fadd double %b, %D
  ret double %c
}
