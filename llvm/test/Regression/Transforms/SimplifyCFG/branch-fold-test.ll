; This test ensures that the simplifycfg pass continues to constant fold
; terminator instructions.

; RUN: llvm-as < %s | opt -simplifycfg | not grep br

int %test(int %A, int %B) {
J:
	%C = add int %A, 12
	br bool true, label %L, label %K  ; K is dead!
L:
	%D = add int %C, %B
	ret int %D
K:
	%E = add int %C, %B
	ret int %E
}
