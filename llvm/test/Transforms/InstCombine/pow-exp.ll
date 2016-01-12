; RUN: opt < %s -instcombine -S | FileCheck %s

define double @pow_exp(double %x, double %y) #0 {
  %call = call fast double @exp(double %x) #0
  %pow = call fast double @llvm.pow.f64(double %call, double %y)
  ret double %pow
}

; CHECK-LABEL: define double @pow_exp(
; CHECK-NEXT:  %mul = fmul fast double %x, %y
; CHECK-NEXT:  %exp = call fast double @exp(double %mul)
; CHECK-NEXT:  ret double %exp

; FIXME: This should not be transformed because the 'exp' call is not fast.
define double @pow_exp_not_fast(double %x, double %y) #0 {
  %call = call double @exp(double %x)
  %pow = call fast double @llvm.pow.f64(double %call, double %y)
  ret double %pow
}

; CHECK-LABEL: define double @pow_exp_not_fast(
; CHECK-NEXT:  %call = call double @exp(double %x)
; CHECK-NEXT:  %mul = fmul fast double %x, %y
; CHECK-NEXT:  %exp = call fast double @exp(double %mul)
; CHECK-NEXT:  ret double %exp

define double @function_pointer(double ()* %fptr, double %p1) #0 {
  %call1 = call fast double %fptr()
  %pow = call fast double @llvm.pow.f64(double %call1, double %p1)
  ret double %pow
}

; CHECK-LABEL: @function_pointer
; CHECK-NEXT:  %call1 = call fast double %fptr()
; CHECK-NEXT:  %pow = call fast double @llvm.pow.f64(double %call1, double %p1)

declare double @exp(double)
declare double @llvm.pow.f64(double, double)
attributes #0 = { "unsafe-fp-math"="true" nounwind readnone }

