; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse

void %test(int %C, <4 x float>* %A, <4 x float>* %B) {
	%tmp = load <4 x float>* %A
	%tmp3 = load <4 x float>* %B
	%tmp9 = mul <4 x float> %tmp3, %tmp3
	%tmp = seteq int %C, 0
	%iftmp.38.0 = select bool %tmp, <4 x float> %tmp9, <4 x float> %tmp
	store <4 x float> %iftmp.38.0, <4 x float>* %A
	ret void
}
