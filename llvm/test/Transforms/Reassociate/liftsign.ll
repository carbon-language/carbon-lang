; RUN: opt -reassociate -gvn -S < %s | FileCheck %s

; (x + 0.1234 * y) * (x + -0.1234 * y) -> (x + 0.1234 * y) * (x - 0.1234 * y)
; so CSE can simplify it further
define double @lift_sign1(double %x, double %y) nounwind readnone ssp uwtable {
; CHECK-LABEL: @lift_sign1(
  %mul = fmul double 1.234000e-01, %y
  %add = fadd double %mul, %x
  %mul1 = fmul double -1.234000e-01, %y
  %add2 = fadd double %mul1, %x
  %mul3 = fmul double %add, %add2
; CHECK-NOT: %mul1 = fmul double -1.234000e-01, %y
; CHECK-NOT: %add2 = fadd %mul1, %x
; CHECK: %add2.repl = fsub double %x, %mul
; CHECK: %mul3 = fmul double %add, %add2
ret double %mul3
}

; (x + -0.1234 * y) * (x + -0.1234 * y) -> (x - 0.1234 * y) * (x - 0.1234 * y)
; GVN can then rewrite it even further
define double @lift_sign2(double %x, double %y) nounwind readnone ssp uwtable {
; CHECK-LABEL: @lift_sign2(
  %mul = fmul double %y, -1.234000e-01
  %add = fadd double %mul, %x
  %mul1 = fmul double %y, -1.234000e-01
  %add2 = fadd double %mul1, %x
  %mul3 = fmul double %add, %add2
; CHECK-NOT: %mul = fmul double %y, -1.234000e-01
; CHECK-NOT: %add = fadd double %mul, %x
; CHECK-NOT: %mul1 = fmul double %y, -1.234000e-01
; CHECK-NOT: %add2 = fadd double %mul1, %x
; CHECK-NOT: %mul3 = fmul double %add, %add2
; CHECK: %mul = fmul double 1.234000e-01, %y
; CHECK: %add.repl = fsub double %x, %mul
; CHECK: %mul3 = fmul double %add.repl, %add.repl
  ret double %mul3
}

; (x + 0.1234 * y) * (x - -0.1234 * y) -> (x + 0.1234 * y) * (x + 0.1234 * y)
define double @lift_sign3(double %x, double %y) nounwind readnone ssp uwtable {
; CHECK-LABEL: @lift_sign3(
  %mul = fmul double %y, 1.234000e-01
  %add = fadd double %mul, %x
  %mul1 = fmul double %y, -1.234000e-01
  %add2 = fsub double %x, %mul1
  %mul3 = fmul double %add, %add2
; CHECK-NOT: %mul1 = fmul double %y, -1.234000e-01
; CHECK-NOT: %add2 = fsub double %x, %mul1
; CHECK-NOT: %mul3 = fmul double %add, %add2
; CHECK: %mul3 = fmul double %add, %add
  ret double %mul3
}

; (x + 0.1234 / y) * (x + -0.1234 / y) -> (x + 0.1234 / y) * (x - 0.1234 / y)
; so CSE can simplify it further
define double @lift_sign4(double %x, double %y) nounwind readnone ssp uwtable {
; CHECK-LABEL: @lift_sign4(
  %div = fdiv double 1.234000e-01, %y
  %add = fadd double %div, %x
  %div1 = fdiv double -1.234000e-01, %y
  %add2 = fadd double %div1, %x
  %mul3 = fmul double %add, %add2
; CHECK-NOT: %div1 = fdiv double -1.234000e-01, %y
; CHECK-NOT: %add2 = fadd double %div1, %x
; CHECK-NOT: %mul3 = fmul double %add, %add2
; CHECK: %add2.repl = fsub double %x, %div
; CHECK: %mul3 = fmul double %add, %add2.repl
  ret double %mul3
}
