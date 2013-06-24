; RUN: llc -mtriple=x86_64-linux -mcpu=nehalem < %s | FileCheck %s --check-prefix=LIN
; RUN: llc -mtriple=x86_64-win32 -mcpu=nehalem < %s | FileCheck %s --check-prefix=WIN
; rdar://7398554

; When doing vector gather-scatter index calculation with 32-bit indices,
; bounce the vector off of cache rather than shuffling each individual
; element out of the index vector.

; CHECK: foo:
; LIN: movaps	(%rsi), %xmm0
; LIN: andps	(%rdx), %xmm0
; LIN: movaps	%xmm0, -24(%rsp)
; LIN: movslq	-24(%rsp), %rsi
; LIN: movslq	-20(%rsp), %rcx
; LIN: movslq	-16(%rsp), %rdx
; LIN: movslq	-12(%rsp), %rax
; LIN: movsd	(%rdi,%rsi,8), %xmm0
; LIN: movhpd	(%rdi,%rcx,8), %xmm0
; LIN: movsd	(%rdi,%rdx,8), %xmm1
; LIN: movhpd	(%rdi,%rax,8), %xmm1

; WIN: movaps	(%rdx), %xmm0
; WIN: andps	(%r8), %xmm0
; WIN: movaps	%xmm0, (%rsp)
; WIN: movslq	(%rsp), %rax
; WIN: movslq	4(%rsp), %rdx
; WIN: movslq	8(%rsp), %r9
; WIN: movslq	12(%rsp), %r8
; WIN: movsd	(%rcx,%rax,8), %xmm0
; WIN: movhpd	(%rcx,%rdx,8), %xmm0
; WIN: movsd	(%rcx,%r9,8), %xmm1
; WIN: movhpd	(%rcx,%r8,8), %xmm1

define <4 x double> @foo(double* %p, <4 x i32>* %i, <4 x i32>* %h) nounwind {
  %a = load <4 x i32>* %i
  %b = load <4 x i32>* %h
  %j = and <4 x i32> %a, %b
  %d0 = extractelement <4 x i32> %j, i32 0
  %d1 = extractelement <4 x i32> %j, i32 1
  %d2 = extractelement <4 x i32> %j, i32 2
  %d3 = extractelement <4 x i32> %j, i32 3
  %q0 = getelementptr double* %p, i32 %d0
  %q1 = getelementptr double* %p, i32 %d1
  %q2 = getelementptr double* %p, i32 %d2
  %q3 = getelementptr double* %p, i32 %d3
  %r0 = load double* %q0
  %r1 = load double* %q1
  %r2 = load double* %q2
  %r3 = load double* %q3
  %v0 = insertelement <4 x double> undef, double %r0, i32 0
  %v1 = insertelement <4 x double> %v0, double %r1, i32 1
  %v2 = insertelement <4 x double> %v1, double %r2, i32 2
  %v3 = insertelement <4 x double> %v2, double %r3, i32 3
  ret <4 x double> %v3
}
