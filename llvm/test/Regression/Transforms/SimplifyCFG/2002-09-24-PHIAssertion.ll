; RUN: llvm-as < %s | opt -simplifycfg

int %test(int %A, int %B, bool %cond) {
J:
	%C = add int %A, 12
	br bool %cond, label %L, label %L
L:
	%Q = phi int [%C, %J], [%C, %J]  ; PHI node is obviously redundant
	%D = add int %C, %B
	%E = add int %Q, %D
	ret int %E
}
