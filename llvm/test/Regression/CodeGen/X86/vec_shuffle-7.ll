; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 &&
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep xorps | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | not grep shufps

void %test() {
	cast <4 x int> zeroinitializer to <4 x float>
	shufflevector <4 x float> %0, <4 x float> zeroinitializer, <4 x uint> zeroinitializer
	store <4 x float> %1, <4 x float>* null
	unreachable
}
