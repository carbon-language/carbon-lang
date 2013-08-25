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

; V8-LABEL:    test_v9_floatreg:
; V8:          fsubd {{.+}}, {{.+}}, {{.+}}
; V8:          faddd {{.+}}, {{.+}}, [[R:%f(((1|2)?(0|2|4|6|8))|30)]]
; V8:          std [[R]], [%{{.+}}]
; V8:          ldd [%{{.+}}], %f0

; V9-LABEL:    test_v9_floatreg:
; V9:          fsubd {{.+}}, {{.+}}, {{.+}}
; V9:          faddd {{.+}}, {{.+}}, [[R:%f((3(2|4|6|8))|((4|5)(0|2|4|6|8))|(60|62))]]
; V9:          fmovd [[R]], %f0


define double @test_v9_floatreg() {
entry:
  %0 = tail call double @get_double()
  %1 = tail call double @get_double()
  %2 = fsub double %0, %1
  tail call void asm sideeffect "", "~{f0},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f18},~{f19},~{f20},~{f21},~{f22},~{f23},~{f24},~{f25},~{f26},~{f27},~{f28},~{f29},~{f30},~{f31}"()
  %3 = fadd double %2, %2
  ret double %3
}


