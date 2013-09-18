; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth=3 -instcombine -gvn -S -mtriple=xcore | FileCheck %s

target datalayout = "e-p:32:32:32-a0:0:32-n32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f16:16:32-f32:32:32-f64:32:32"
target triple = "xcore"

; Basic depth-3 chain
define double @test1(double %A1, double %A2, double %B1, double %B2) {
; CHECK-LABEL: @test1(
; CHECK-NOT: <2 x double>
  %X1 = fsub double %A1, %B1
  %X2 = fsub double %A2, %B2
  %Y1 = fmul double %X1, %A1
  %Y2 = fmul double %X2, %A2
  %Z1 = fadd double %Y1, %B1
  %Z2 = fadd double %Y2, %B2
  %R  = fmul double %Z1, %Z2
  ret double %R
}
