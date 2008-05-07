; RUN: llvm-as < %s | llc -march=x86 -mattr=+mmx | grep movq | count 3

; FIXME: This code outputs:
;
;   subl $28, %esp
;   movl 32(%esp), %eax
;   movd %eax, %mm0
;   movq %mm0, (%esp)
;   movl (%esp), %eax
;   movl %eax, 20(%esp)
;   movq %mm0, 8(%esp)
;   movl 12(%esp), %eax
;   movl %eax, 16(%esp)
;   movq 16(%esp), %mm0
;   addl $28, %esp
;
; Which is ugly. We need to fix this.

define <2 x i32> @qux(i32 %A) nounwind {
entry:
	%tmp3 = insertelement <2 x i32> < i32 0, i32 undef >, i32 %A, i32 1		; <<2 x i32>> [#uses=1]
	ret <2 x i32> %tmp3
}
