; RUN: opt < %s -instcombine -S | FileCheck %s

define double @mypow(double %x, double %y) #0 {
entry:
  %call = call double @exp(double %x)
  %pow = call double @llvm.pow.f64(double %call, double %y)
  ret double %pow
}

; CHECK-LABEL: define double @mypow(
; CHECK:   %call = call double @exp(double %x)
; CHECK:   %pow = call double @llvm.pow.f64(double %call, double %y)
; CHECK:   ret double %pow
; CHECK: }

declare double @exp(double) #1
declare double @llvm.pow.f64(double, double)
