; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine -gcse -instcombine | \
; RUN:    llvm-dis | not grep getelementptr

bool %test(int* %A) {
	%B = getelementptr int* %A, int 1
	%C = getelementptr int* %A, uint 1
	%V = seteq int* %B, %C
	ret bool %V
}
