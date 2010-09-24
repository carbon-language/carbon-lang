; RUN: opt -instcombine -S < %s | FileCheck %s

; This test is inexplicably still failing, which suggests a bug in the host
; libm. It appears that sin(inf) returns NaN without setting a floating point
; exception.
; XFAIL: arm-pc-linux-gnu

; This shouldn't fold, because sin(inf) is invalid.
; CHECK: @foo
; CHECK:   %t = call double @sin(double 0x7FF0000000000000)
define double @foo() {
  %t = call double @sin(double 0x7FF0000000000000)
  ret double %t
}

; This should fold.
; CHECK: @bar
; CHECK:   ret double 0.0
define double @bar() {
  %t = call double @sin(double 0.0)
  ret double %t
}

declare double @sin(double)
