; RUN: llvm-upgrade < %s | llvm-as | opt -sccp | llvm-dis | \
; RUN:   grep {ret i1 false}

bool %foo() {
	%X = and bool false, undef
	ret bool %X
}
