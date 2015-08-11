; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=sse2 | FileCheck %s
; Ideally this would compile to 5 multiplies.

define double @pow_wrapper(double %a) nounwind readonly ssp noredzone {
; CHECK-LABEL: pow_wrapper:
; CHECK:       # BB#0:
; CHECK-NEXT:    movapd %xmm0, %xmm1
; CHECK-NEXT:    mulsd %xmm1, %xmm1
; CHECK-NEXT:    mulsd %xmm1, %xmm0
; CHECK-NEXT:    mulsd %xmm1, %xmm1
; CHECK-NEXT:    mulsd %xmm1, %xmm0
; CHECK-NEXT:    mulsd %xmm1, %xmm1
; CHECK-NEXT:    mulsd %xmm0, %xmm1
; CHECK-NEXT:    movapd %xmm1, %xmm0
; CHECK-NEXT:    retq
  %ret = tail call double @llvm.powi.f64(double %a, i32 15) nounwind ; <double> [#uses=1]
  ret double %ret
}

declare double @llvm.powi.f64(double, i32) nounwind readonly

