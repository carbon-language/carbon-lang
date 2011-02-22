; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s
; RUN: llc -O0 < %s -march=x86 -mcpu=core2 | FileCheck %s --check-prefix=CHECK_O0

define <4 x i32> @t00(<4 x i32>* %a0) nounwind ssp {
entry:
; CHECK: movaps  ({{%rdi|%rcx}}), %xmm0
; CHECK-NEXT: movaps  %xmm0, %xmm1
; CHECK-NEXT: movlps  (%rax), %xmm1
; CHECK-NEXT: shufps  $36, %xmm1, %xmm0
  %0 = load <4 x i32>* undef, align 16
  %1 = load <4 x i32>* %a0, align 16
  %2 = shufflevector <4 x i32> %1, <4 x i32> %0, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x i32> %2
}

define void @t01(double* %a0) nounwind ssp {
entry:
; CHECK_O0: movsd (%eax), %xmm0
; CHECK_O0: unpcklpd  %xmm0, %xmm0
  %tmp93 = load double* %a0, align 8
  %vecinit94 = insertelement <2 x double> undef, double %tmp93, i32 1
  store <2 x double> %vecinit94, <2 x double>* undef
  ret void
}
