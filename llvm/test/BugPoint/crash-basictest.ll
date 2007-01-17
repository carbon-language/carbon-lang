; Basic test for bugpoint.
; RUN: llvm-upgrade < %s > %t1.ll
; RUN: bugpoint %t1.ll -domset -idom -domset -bugpoint-crashcalls \
; RUN:   -domset -idom -domset

int %test() {
	call int %test()
	ret int %0
}
