; RUN: llc -mtriple=x86_64-apple-darwin -fast-isel -code-model=small < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-darwin -fast-isel -code-model=large < %s | FileCheck %s --check-prefix=LARGE

; Make sure fast isel uses rip-relative addressing for the small code model.
define float @constpool_float(float %x) {
; CHECK-LABEL: constpool_float
; CHECK:       LCPI0_0(%rip)

; LARGE-LABEL: constpool_float
; LARGE:       movabsq  $LCPI0_0, %rax
  %1 = fadd float %x, 16.50e+01
  ret float %1
}

define double @constpool_double(double %x) nounwind {
; CHECK-LABEL: constpool_double
; CHECK:       LCPI1_0(%rip)

; LARGE-LABEL: constpool_double
; LARGE:       movabsq  $LCPI1_0, %rax
  %1 = fadd double %x, 8.500000e-01
  ret double %1
}
