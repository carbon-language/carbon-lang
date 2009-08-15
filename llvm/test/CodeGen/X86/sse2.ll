; Tests for SSE2 and below, without SSE3+.
; RUN: llvm-as < %s | llc -march=x86 -mcpu=pentium4 | FileCheck %s

define void @t1(<2 x double>* %r, <2 x double>* %A, double %B) nounwind  {
	%tmp3 = load <2 x double>* %A, align 16
	%tmp7 = insertelement <2 x double> undef, double %B, i32 0
	%tmp9 = shufflevector <2 x double> %tmp3, <2 x double> %tmp7, <2 x i32> < i32 2, i32 1 >
	store <2 x double> %tmp9, <2 x double>* %r, align 16
	ret void
        
; CHECK: t1:
; CHECK: 	movl	8(%esp), %eax
; CHECK: 	movapd	(%eax), %xmm0
; CHECK: 	movlpd	12(%esp), %xmm0
; CHECK: 	movl	4(%esp), %eax
; CHECK: 	movapd	%xmm0, (%eax)
; CHECK: 	ret
}

define void @t2(<2 x double>* %r, <2 x double>* %A, double %B) nounwind  {
	%tmp3 = load <2 x double>* %A, align 16
	%tmp7 = insertelement <2 x double> undef, double %B, i32 0
	%tmp9 = shufflevector <2 x double> %tmp3, <2 x double> %tmp7, <2 x i32> < i32 0, i32 2 >
	store <2 x double> %tmp9, <2 x double>* %r, align 16
	ret void
        
; CHECK: t2:
; CHECK: 	movl	8(%esp), %eax
; CHECK: 	movapd	(%eax), %xmm0
; CHECK: 	movhpd	12(%esp), %xmm0
; CHECK: 	movl	4(%esp), %eax
; CHECK: 	movapd	%xmm0, (%eax)
; CHECK: 	ret
}
