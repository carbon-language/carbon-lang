; RUN: bugpoint %s  -bugpoint-crashcalls

; Test to make sure that arguments are removed from the function if they are unnecessary.

declare int %test2()
int %test(int %A, int %B, float %C) {
	call int %test2()
	ret int %0
}
