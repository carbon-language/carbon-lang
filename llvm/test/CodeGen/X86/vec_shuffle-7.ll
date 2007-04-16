; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -mattr=+sse2 -o %t  -f
; RUN: grep xorps %t | wc -l | grep 1
; RUN: not grep shufps %t

void %test() {
	cast <4 x int> zeroinitializer to <4 x float>
	shufflevector <4 x float> %0, <4 x float> zeroinitializer, <4 x uint> zeroinitializer
	store <4 x float> %1, <4 x float>* null
	unreachable
}
