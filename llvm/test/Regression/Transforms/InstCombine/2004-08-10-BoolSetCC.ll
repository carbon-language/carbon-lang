; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep 'ret bool false'
bool %test(bool %V) {
	%Y = setlt bool %V, false
	ret bool %Y
}

