; RUN: llc < %s -march=x86-64 > %t
; RUN: grep subq %t | count 1
; RUN: grep addq %t | count 1

define x86_fp80 @f0(float %f) nounwind readnone noredzone {
entry:
	%0 = fpext float %f to x86_fp80		; <x86_fp80> [#uses=1]
	ret x86_fp80 %0
}
