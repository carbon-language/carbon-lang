; RUN: llc -march=sparc < %s | FileCheck %s -check-prefix=V8
; RUN: llc -march=sparc -O0 < %s | FileCheck %s -check-prefix=V8-UNOPT
; RUN: llc -march=sparc -mattr=v9 < %s | FileCheck %s -check-prefix=V9


; V8-LABEL:     test_neg:
; V8:     call get_double
; V8:     fnegs %f0, %f0

; V8-UNOPT-LABEL:     test_neg:
; V8-UNOPT:     fnegs
; V8-UNOPT:     ! implicit-def
; V8-UNOPT:     fmovs {{.+}}, %f0
; V8-UNOPT:     fmovs {{.+}}, %f1

; V9-LABEL:     test_neg:
; V9:     fnegd %f0, %f0

define double @test_neg() {
entry:
  %0 = tail call double @get_double()
  %1 = fsub double -0.000000e+00, %0
  ret double %1
}

; V8-LABEL:     test_abs:
; V8:     fabss %f0, %f0

; V8-UNOPT-LABEL:     test_abs:
; V8-UNOPT:     fabss
; V8-UNOPT:     ! implicit-def
; V8-UNOPT:     fmovs {{.+}}, %f0
; V8-UNOPT:     fmovs {{.+}}, %f1

; V9-LABEL:     test_abs:
; V9:     fabsd %f0, %f0

define double @test_abs() {
entry:
  %0 = tail call double @get_double()
  %1 = tail call double @llvm.fabs.f64(double %0)
  ret double %1
}

declare double @get_double()
declare double @llvm.fabs.f64(double) nounwind readonly

