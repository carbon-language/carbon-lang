; RUN: llc < %s -march=sparc -mattr=hard-quad-float | FileCheck %s

; CHECK-LABEL: f128_ops
; CHECK:       ldd
; CHECK:       ldd
; CHECK:       ldd
; CHECK:       ldd
; CHECK:       faddq [[R0:.+]],  [[R1:.+]],  [[R2:.+]]
; CHECK:       fsubq [[R2]], [[R3:.+]], [[R4:.+]]
; CHECK:       fmulq [[R4]], [[R5:.+]], [[R6:.+]]
; CHECK:       fdivq [[R6]], [[R2]]
; CHECK:       std
; CHECK:       std

define void @f128_ops(fp128* noalias sret %scalar.result, fp128* byval %a, fp128* byval %b, fp128* byval %c, fp128* byval %d) {
entry:
  %0 = load fp128* %a, align 8
  %1 = load fp128* %b, align 8
  %2 = load fp128* %c, align 8
  %3 = load fp128* %d, align 8
  %4 = fadd fp128 %0, %1
  %5 = fsub fp128 %4, %2
  %6 = fmul fp128 %5, %3
  %7 = fdiv fp128 %6, %4
  store fp128 %7, fp128* %scalar.result, align 8
  ret void
}
