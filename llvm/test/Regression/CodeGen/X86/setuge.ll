; RUN: llvm-as < %s | llc -march=x86 | not grep 'set'

declare bool %llvm.isunordered.f32(float, float)

float %cmp(float %A, float %B, float %C, float %D) {
entry:
	%tmp.1 = call bool %llvm.isunordered.f32(float %A, float %B)
	%tmp.2 = setge float %A, %B
	%tmp.3 = or bool %tmp.1, %tmp.2
	%tmp.4 = select bool %tmp.3, float %C, float %D
	ret float %tmp.4
}
