; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 &&
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | not grep and
<4 x float> %test(<4 x float>* %v1) {
	%tmp = load <4 x float>* %v1
	%tmp15 = cast <4 x float> %tmp to <2 x long>
	%tmp24 = and <2 x long> %tmp15, cast (<4 x int> < int 0, int 0, int -1, int -1 > to <2 x long>)
	%tmp31 = cast <2 x long> %tmp24 to <4 x float>
	ret <4 x float> %tmp31
}
