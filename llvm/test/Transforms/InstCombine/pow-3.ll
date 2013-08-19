; Test that the pow won't get simplified to sqrt(fabs) when they are not available.
;
; RUN: opt < %s -disable-simplify-libcalls -instcombine -S | FileCheck %s

declare double @llvm.pow.f64(double %Val, double %Power)

define double @test_simplify_unavailable(double %x) {
; CHECK-LABEL: @test_simplify_unavailable(
  %retval = call double @llvm.pow.f64(double %x, double 0.5)
; CHECK-NEXT: call double @llvm.pow.f64(double %x, double 5.000000e-01)
  ret double %retval
}
