; RUN: llvm-as < %s | opt -instcombine -gcse -instcombine | llvm-dis | not grep getelementptr

bool %test(int* %A) {
	%B = getelementptr int* %A, int 1
	%C = getelementptr int* %A, uint 1
	%V = seteq int* %B, %C
	ret bool %V
}
