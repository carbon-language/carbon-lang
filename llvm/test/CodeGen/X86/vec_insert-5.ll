; RUN: llc < %s -march=x86 -mattr=+sse2 | FileCheck %s
; There are no MMX operations in @t1

define void  @t1(i32 %a, x86_mmx* %P) nounwind {
       %tmp12 = shl i32 %a, 12
       %tmp21 = insertelement <2 x i32> undef, i32 %tmp12, i32 1
       %tmp22 = insertelement <2 x i32> %tmp21, i32 0, i32 0
       %tmp23 = bitcast <2 x i32> %tmp22 to x86_mmx
       store x86_mmx %tmp23, x86_mmx* %P
       ret void

; CHECK: t1:
; CHECK-NOT: %mm
; CHECK: shll $12
; CHECK-NOT: %mm
}

define <4 x float> @t2(<4 x float>* %P) nounwind {
        %tmp1 = load <4 x float>* %P
        %tmp2 = shufflevector <4 x float> %tmp1, <4 x float> zeroinitializer, <4 x i32> < i32 4, i32 4, i32 4, i32 0 >
        ret <4 x float> %tmp2

; CHECK: t2:
; CHECK: pslldq $12
}

define <4 x float> @t3(<4 x float>* %P) nounwind {
        %tmp1 = load <4 x float>* %P
        %tmp2 = shufflevector <4 x float> %tmp1, <4 x float> zeroinitializer, <4 x i32> < i32 2, i32 3, i32 4, i32 4 >
        ret <4 x float> %tmp2

; CHECK: t3:
; CHECK: psrldq $8
}

define <4 x float> @t4(<4 x float>* %P) nounwind {
        %tmp1 = load <4 x float>* %P
        %tmp2 = shufflevector <4 x float> zeroinitializer, <4 x float> %tmp1, <4 x i32> < i32 7, i32 0, i32 0, i32 0 >
        ret <4 x float> %tmp2

; CHECK: t4:
; CHECK: psrldq $12
}
