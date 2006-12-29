; RUN: llvm-upgrade < %s | llvm-as | opt -sccp | llvm-dis | \
; RUN:   grep 'ret bool false'

bool %foo() {
	%X = and bool false, undef
	ret bool %X
}
