; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:    grep {ret i1 false}
bool %test(bool %V) {
	%Y = setlt bool %V, false
	ret bool %Y
}

