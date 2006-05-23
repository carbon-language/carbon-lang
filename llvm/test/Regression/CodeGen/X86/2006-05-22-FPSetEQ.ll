; RUN: llvm-as < %s | llc -march=x86 &&
; RUN: llvm-as < %s | llc -march=x86 | grep 'setnp'
; RUN: llvm-as < %s | llc -march=x86 -enable-unsafe-fp-math | not grep 'setnp'

uint %test(float %f) {
	%tmp = seteq float %f, 0.000000e+00
	%tmp = cast bool %tmp to uint
	ret uint %tmp
}
