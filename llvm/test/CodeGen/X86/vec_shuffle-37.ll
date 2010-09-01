; RUN: llc < %s -march=x86-64 | FileCheck %s

define <4 x i32> @t00(<4 x i32>* %a0) nounwind ssp {
entry:
; CHECK: movaps  (%rdi), %xmm0
; CHECK-NEXT: movaps  %xmm0, %xmm1
; CHECK-NEXT: movlps  (%rax), %xmm1
; CHECK-NEXT: shufps  $36, %xmm1, %xmm0
  %0 = load <4 x i32>* undef, align 16
  %1 = load <4 x i32>* %a0, align 16
  %2 = shufflevector <4 x i32> %1, <4 x i32> %0, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x i32> %2
}

