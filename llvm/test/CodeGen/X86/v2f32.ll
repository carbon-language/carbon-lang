; RUN: llc < %s -march=x86-64 -asm-verbose=0 -o - | FileCheck %s

; PR7518
define void @test1(<2 x float> %Q, float *%P2) nounwind {
  %a = extractelement <2 x float> %Q, i32 0
  %b = extractelement <2 x float> %Q, i32 1
  %c = fadd float %a, %b

  store float %c, float* %P2
  ret void
; CHECK: test1:
; CHECK-NEXT: addss	%xmm1, %xmm0
; CHECK-NEXT: movss	%xmm0, (%rdi)
; CHECK-NEXT: ret
}

