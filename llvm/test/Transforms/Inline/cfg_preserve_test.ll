; This test ensures that inlining an "empty" function does not destroy the CFG
;
; RUN: llvm-upgrade < %s | llvm-as | opt -inline | llvm-dis | not grep br

int %func(int %i) {
	ret int %i
}

declare void %bar()

int %main(int %argc) {
Entry:
	%X = call int %func(int 7)
	ret int %X
}
