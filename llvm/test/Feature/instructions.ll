; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

uint %test_extractelement(<4 x uint> %V) {
	%R = extractelement <4 x uint> %V, uint 1
	ret uint %R
}
