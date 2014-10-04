; RUN: llc < %s -mcpu=corei7 -march=x86 -mattr=+sse2,-sse4.1 | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define void @test1(<4 x float>* %F, float* %f) nounwind {
; CHECK-LABEL: test1:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; CHECK-NEXT:    movaps (%ecx), %xmm0
; CHECK-NEXT:    addps %xmm0, %xmm0
; CHECK-NEXT:    movss %xmm0, (%eax)
; CHECK-NEXT:    retl
entry:
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=2]
	%tmp7 = fadd <4 x float> %tmp, %tmp		; <<4 x float>> [#uses=1]
	%tmp2 = extractelement <4 x float> %tmp7, i32 0		; <float> [#uses=1]
	store float %tmp2, float* %f
	ret void
}

define float @test2(<4 x float>* %F, float* %f) nounwind {
; CHECK-LABEL: test2:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    pushl %eax
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    movaps (%eax), %xmm0
; CHECK-NEXT:    addps %xmm0, %xmm0
; CHECK-NEXT:    shufpd {{.*#+}} xmm0 = xmm0[1,0]
; CHECK-NEXT:    movss %xmm0, (%esp)
; CHECK-NEXT:    flds (%esp)
; CHECK-NEXT:    popl %eax
; CHECK-NEXT:    retl
entry:
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=2]
	%tmp7 = fadd <4 x float> %tmp, %tmp		; <<4 x float>> [#uses=1]
	%tmp2 = extractelement <4 x float> %tmp7, i32 2		; <float> [#uses=1]
	ret float %tmp2
}

define void @test3(float* %R, <4 x float>* %P1) nounwind {
; CHECK-LABEL: test3:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; CHECK-NEXT:    movss 12(%ecx), %xmm0
; CHECK-NEXT:    movss %xmm0, (%eax)
; CHECK-NEXT:    retl
entry:
	%X = load <4 x float>* %P1		; <<4 x float>> [#uses=1]
	%tmp = extractelement <4 x float> %X, i32 3		; <float> [#uses=1]
	store float %tmp, float* %R
	ret void
}

define double @test4(double %A) nounwind {
; CHECK-LABEL: test4:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    subl $12, %esp
; CHECK-NEXT:    calll foo
; CHECK-NEXT:    shufpd {{.*#+}} xmm0 = xmm0[1,0]
; CHECK-NEXT:    addsd {{[0-9]+}}(%esp), %xmm0
; CHECK-NEXT:    movsd %xmm0, (%esp)
; CHECK-NEXT:    fldl (%esp)
; CHECK-NEXT:    addl $12, %esp
; CHECK-NEXT:    retl
entry:
	%tmp1 = call <2 x double> @foo( )		; <<2 x double>> [#uses=1]
	%tmp2 = extractelement <2 x double> %tmp1, i32 1		; <double> [#uses=1]
	%tmp3 = fadd double %tmp2, %A		; <double> [#uses=1]
	ret double %tmp3
}

declare <2 x double> @foo()
