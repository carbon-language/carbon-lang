; Basic test for bugpoint.
; RUN: bugpoint %s -domset -idom -domset -bugpoint-crashcalls -domset -idom -domset

int %test() {
	call int %test()
	ret int %0
}
