; RUN: llc < %s -mtriple=i686-darwin -mattr=+mmx,+sse2 | FileCheck %s -check-prefix=X86-32
; RUN: llc < %s -mtriple=x86_64-darwin -mattr=+mmx,+sse4.1 | FileCheck %s -check-prefix=X86-64

; This is not an MMX operation; promoted to XMM.
define x86_mmx @t0(i32 %A) nounwind {
; X86-32-LABEL: t0:
; X86-32:       ## BB#0:
; X86-32:    movd {{[0-9]+}}(%esp), %xmm0
; X86-32-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[1,0,0,1]
; X86-32-NEXT:    movlpd %xmm0, (%esp)
; X86-32-NEXT:    movq (%esp), %mm0
; X86-32-NEXT:    addl $12, %esp
; X86-32-NEXT:    retl
  %tmp3 = insertelement <2 x i32> < i32 0, i32 undef >, i32 %A, i32 1
  %tmp4 = bitcast <2 x i32> %tmp3 to x86_mmx
  ret x86_mmx %tmp4
}

define <8 x i8> @t1(i8 zeroext %x) nounwind {
; X86-32-LABEL: t1:
; X86-32:       ## BB#0:
; X86-32-NOT:  movl
; X86-32-NEXT:    movd {{[0-9]+}}(%esp), %xmm0
; X86-32-NEXT:    retl
  %r = insertelement <8 x i8> undef, i8 %x, i32 0
  ret <8 x i8> %r
}

; PR2574
define <2 x float> @t2(<2 x float> %a0) {
; X86-32-LABEL: t2:
; X86-32:       ## BB#0:
; X86-32-NEXT:    xorps %xmm0, %xmm0
; X86-32-NEXT:    retl
  %v1 = insertelement <2 x float> %a0, float 0.000000e+00, i32 0
  %v2 = insertelement <2 x float> %v1, float 0.000000e+00, i32 1
  ret <2 x float> %v2
}

@g0 = external global i16
@g1 = external global <4 x i16>

; PR2562
define void @t3() {
; X86-64-LABEL: t3:
; X86-64:       ## BB#0:
; X86-64:    pmovzxwd (%rcx)
; X86-64-NEXT:    movzwl
; X86-64-NEXT:    pinsrd $0
; X86-64-NEXT:    pshufb {{.*#+}} xmm0 = xmm0[0,1,4,5,8,9,12,13,8,9,12,13,12,13,14,15]
; X86-64-NEXT:    movq %xmm0
; X86-64-NEXT:    retq
  load i16, i16* @g0
  load <4 x i16>, <4 x i16>* @g1
  insertelement <4 x i16> %2, i16 %1, i32 0
  store <4 x i16> %3, <4 x i16>* @g1
  ret void
}
