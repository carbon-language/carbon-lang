; RUN: llc < %s -mcpu=corei7 -march=x86 -mattr=+sse4.1 | FileCheck %s

define void @t1(float* %R, <4 x float>* %P1) nounwind {
; CHECK-LABEL: @t1
; CHECK:         movl 4(%esp), %[[R0:e[abcd]x]]
; CHECK-NEXT:    movl 8(%esp), %[[R1:e[abcd]x]]
; CHECK-NEXT:    movss 12(%[[R1]]), %[[R2:xmm.*]]
; CHECK-NEXT:    movss %[[R2]], (%[[R0]])
; CHECK-NEXT:    retl

	%X = load <4 x float>* %P1
	%tmp = extractelement <4 x float> %X, i32 3
	store float %tmp, float* %R
	ret void
}

define float @t2(<4 x float>* %P1) nounwind {
; CHECK-LABEL: @t2
; CHECK:         movl 4(%esp), %[[R0:e[abcd]x]]
; CHECK-NEXT:    flds 8(%[[R0]])
; CHECK-NEXT:    retl

	%X = load <4 x float>* %P1
	%tmp = extractelement <4 x float> %X, i32 2
	ret float %tmp
}

define void @t3(i32* %R, <4 x i32>* %P1) nounwind {
; CHECK-LABEL: @t3
; CHECK:         movl 4(%esp), %[[R0:e[abcd]x]]
; CHECK-NEXT:    movl 8(%esp), %[[R1:e[abcd]x]]
; CHECK-NEXT:    movl 12(%[[R1]]), %[[R2:e[abcd]x]]
; CHECK-NEXT:    movl %[[R2]], (%[[R0]])
; CHECK-NEXT:    retl

	%X = load <4 x i32>* %P1
	%tmp = extractelement <4 x i32> %X, i32 3
	store i32 %tmp, i32* %R
	ret void
}

define i32 @t4(<4 x i32>* %P1) nounwind {
; CHECK-LABEL: @t4
; CHECK:         movl 4(%esp), %[[R0:e[abcd]x]]
; CHECK-NEXT:    movl 12(%[[R0]]), %eax
; CHECK-NEXT:    retl

	%X = load <4 x i32>* %P1
	%tmp = extractelement <4 x i32> %X, i32 3
	ret i32 %tmp
}
