; RUN: llc < %s -march=x86-64 -mcpu=penryn -asm-verbose=0 -o - | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mcpu=yonah -march=x86 -asm-verbose=0 -o - | FileCheck %s -check-prefix=X32

; PR7518
define void @test1(<2 x float> %Q, float *%P2) nounwind {
  %a = extractelement <2 x float> %Q, i32 0
  %b = extractelement <2 x float> %Q, i32 1
  %c = fadd float %a, %b

  store float %c, float* %P2
  ret void
; X64: test1:
; X64-NEXT: addss	%xmm1, %xmm0
; X64-NEXT: movss	%xmm0, (%rdi)
; X64-NEXT: ret

; X32: test1:
; X32-NEXT: movss	4(%esp), %xmm0
; X32-NEXT: addss	8(%esp), %xmm0
; X32-NEXT: movl	12(%esp), %eax
; X32-NEXT: movss	%xmm0, (%eax)
; X32-NEXT: ret
}


define <2 x float> @test2(<2 x float> %Q, <2 x float> %R, <2 x float> *%P) nounwind {
  %Z = fadd <2 x float> %Q, %R
  ret <2 x float> %Z
  
; X64: test2:
; X64-NEXT: insertps $0
; X64-NEXT: insertps $16
; X64-NEXT: insertps $0
; X64-NEXT: insertps $16
; X64-NEXT: addps
; X64-NEXT: movaps
; X64-NEXT: pshufd
; X64-NEXT: ret
}
