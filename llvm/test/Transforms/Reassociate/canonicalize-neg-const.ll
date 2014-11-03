; RUN: opt -reassociate -gvn -S < %s | FileCheck %s

; (x + 0.1234 * y) * (x + -0.1234 * y) -> (x + 0.1234 * y) * (x - 0.1234 * y)
define double @test1(double %x, double %y) {
; CHECK-LABEL: @test1
; CHECK-NEXT: fmul double 1.234000e-01, %y
; CHECK-NEXT: fadd double %x, %mul
; CHECK-NEXT: fsub double %x, %mul
; CHECK-NEXT: fmul double %add{{.*}}, %add{{.*}}
; CHECK-NEXT: ret double %mul

  %mul = fmul double 1.234000e-01, %y
  %add = fadd double %mul, %x
  %mul1 = fmul double -1.234000e-01, %y
  %add2 = fadd double %mul1, %x
  %mul3 = fmul double %add, %add2
  ret double %mul3
}

; (x + -0.1234 * y) * (x + -0.1234 * y) -> (x - 0.1234 * y) * (x - 0.1234 * y)
define double @test2(double %x, double %y) {
; CHECK-LABEL: @test2
; CHECK-NEXT: fmul double 1.234000e-01, %y
; CHECK-NEXT: fsub double %x, %mul
; CHECK-NEXT: fmul double %add{{.*}}, %add{{.*}}
; CHECK-NEXT: ret double %mul

  %mul = fmul double %y, -1.234000e-01
  %add = fadd double %mul, %x
  %mul1 = fmul double %y, -1.234000e-01
  %add2 = fadd double %mul1, %x
  %mul3 = fmul double %add, %add2
  ret double %mul3
}

; (x + 0.1234 / y) * (x + -0.1234 / y) -> (x + 0.1234 / y) * (x - 0.1234 / y)
define double @test3(double %x, double %y) {
; CHECK-LABEL: @test3
; CHECK-NEXT: fdiv double 1.234000e-01, %y
; CHECK-NEXT: fadd double %x, %div
; CHECK-NEXT: fsub double %x, %div
; CHECK-NEXT: fmul double %add{{.*}}, %add{{.*}}
; CHECK-NEXT: ret double

  %div = fdiv double 1.234000e-01, %y
  %add = fadd double %div, %x
  %div1 = fdiv double -1.234000e-01, %y
  %add2 = fadd double %div1, %x
  %mul3 = fmul double %add, %add2
  ret double %mul3
}

; (x + 0.1234 * y) * (x - -0.1234 * y) -> (x + 0.1234 * y) * (x + 0.1234 * y)
define double @test4(double %x, double %y) {
; CHECK-LABEL: @test4
; CHECK-NEXT: fmul double 1.234000e-01, %y
; CHECK-NEXT: fadd double %x, %mul
; CHECK-NEXT: fmul double %add{{.*}}, %add{{.*}}
; CHECK-NEXT: ret double

  %mul = fmul double %y, 1.234000e-01
  %add = fadd double %mul, %x
  %mul1 = fmul double %y, -1.234000e-01
  %add2 = fsub double %x, %mul1
  %mul3 = fmul double %add, %add2
  ret double %mul3
}

; Canonicalize (x - -1234 * y)
define i64 @test5(i64 %x, i64 %y) {
; CHECK-LABEL: @test5
; CHECK-NEXT: mul i64 %y, 1234
; CHECK-NEXT: add i64 %mul, %x
; CHECK-NEXT: ret i64 %sub

  %mul = mul i64 %y, -1234
  %sub = sub i64 %x, %mul
  ret i64 %sub
}

; Canonicalize (x - -0.1234 * y)
define double @test6(double %x, double %y) {
; CHECK-LABEL: @test6
; CHECK-NEXT: fmul double 1.234000e-01, %y
; CHECK-NEXT: fadd double %x, %mul
; CHECK-NEXT: ret double

  %mul = fmul double -1.234000e-01, %y
  %sub = fsub double %x, %mul
  ret double %sub
}

; Don't modify (-0.1234 * y - x)
define double @test7(double %x, double %y) {
; CHECK-LABEL: @test7
; CHECK-NEXT: fmul double -1.234000e-01, %y
; CHECK-NEXT: fsub double %mul, %x
; CHECK-NEXT: ret double %sub

  %mul = fmul double -1.234000e-01, %y
  %sub = fsub double %mul, %x
  ret double %sub
}

; Canonicalize (-0.1234 * y + x) -> (x - 0.1234 * y)
define double @test8(double %x, double %y) {
; CHECK-LABEL: @test8
; CHECK-NEXT: fmul double 1.234000e-01, %y
; CHECK-NEXT: fsub double %x, %mul
; CHECK-NEXT: ret double %add

  %mul = fmul double -1.234000e-01, %y
  %add = fadd double %mul, %x
  ret double %add
}

; Canonicalize (y * -0.1234 + x) -> (x - 0.1234 * y)
define double @test9(double %x, double %y) {
; CHECK-LABEL: @test9
; CHECK-NEXT: fmul double 1.234000e-01, %y
; CHECK-NEXT: fsub double %x, %mul
; CHECK-NEXT: ret double %add

  %mul = fmul double %y, -1.234000e-01
  %add = fadd double %mul, %x
  ret double %add
}

; Canonicalize (x - -1234 udiv y)
define i64 @test10(i64 %x, i64 %y) {
; CHECK-LABEL: @test10
; CHECK-NEXT: udiv i64 1234, %y
; CHECK-NEXT: add i64 %div, %x
; CHECK-NEXT: ret i64 %sub

  %div = udiv i64 -1234, %y
  %sub = sub i64 %x, %div
  ret i64 %sub
}

; Canonicalize (x - -1234 sdiv y)
define i64 @test11(i64 %x, i64 %y) {
; CHECK-LABEL: @test11
; CHECK-NEXT: sdiv i64 1234, %y
; CHECK-NEXT: add i64 %div, %x
; CHECK-NEXT: ret i64 %sub

  %div = sdiv i64 -1234, %y
  %sub = sub i64 %x, %div
  ret i64 %sub
}

; Canonicalize (x - -0.1234 / y)
define double @test12(double %x, double %y) {
; CHECK-LABEL: @test12
; CHECK-NEXT: fdiv double 1.234000e-01, %y
; CHECK-NEXT: fadd double %x, %div
; CHECK-NEXT: ret double %sub

  %div = fdiv double -1.234000e-01, %y
  %sub = fsub double %x, %div
  ret double %sub
}

; Canonicalize (x - -0.1234 urem y)
define i64 @test13(i64 %x, i64 %y) {
; CHECK-LABEL: @test13
; CHECK-NEXT: urem i64 1234, %y
; CHECK-NEXT: add i64 %urem, %x
; CHECK-NEXT: ret i64 %sub

  %urem = urem i64 -1234, %y
  %sub = sub i64 %x, %urem
  ret i64 %sub
}

; Canonicalize (x + -0.1234 srem y)
define i64 @test14(i64 %x, i64 %y) {
; CHECK-LABEL: @test14
; CHECK-NEXT: srem i64 1234, %y
; CHECK-NEXT: sub i64 %x, %srem
; CHECK-NEXT: ret i64 %add

  %srem = srem i64 -1234, %y
  %add = add i64 %x, %srem
  ret i64 %add
}

; Don't modify (-0.1234 srem y - x)
define i64 @test15(i64 %x, i64 %y) {
; CHECK-LABEL: @test15
; CHECK-NEXT: %srem = srem i64 -1234, %y
; CHECK-NEXT: %sub = sub i64 %srem, %x
; CHECK-NEXT: ret i64 %sub

  %srem = srem i64 -1234, %y
  %sub = sub i64 %srem, %x
  ret i64 %sub
}

; Canonicalize (x - -0.1234 frem y)
define double @test16(double %x, double %y) {
; CHECK-LABEL: @test16
; CHECK-NEXT: frem double 1.234000e-01, %y
; CHECK-NEXT: fadd double %x, %rem
; CHECK-NEXT: ret double %sub

  %rem = frem double -1.234000e-01, %y
  %sub = fsub double %x, %rem
  ret double %sub
}

; Canonicalize (x + -0.1234 frem y)
define double @test17(double %x, double %y) {
; CHECK-LABEL: @test17
; CHECK-NEXT: frem double 1.234000e-01, %y
; CHECK-NEXT: fsub double %x, %rem
; CHECK-NEXT: ret double %add

  %rem = frem double -1.234000e-01, %y
  %add = fadd double %x, %rem
  ret double %add
}

; Canonicalize (-0.1234 frem y + x) -> (x - 0.1234 frem y)
define double @test18(double %x, double %y) {
; CHECK-LABEL: @test18
; CHECK-NEXT: frem double 1.234000e-01, %y
; CHECK-NEXT: fsub double %x, %rem
; CHECK-NEXT: ret double %add

  %rem = frem double -1.234000e-01, %y
  %add = fadd double %x, %rem
  ret double %add
}
