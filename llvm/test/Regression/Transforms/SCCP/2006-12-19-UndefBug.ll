; RUN: llvm-as < %s | opt -sccp | llvm-dis | grep 'ret bool false'

bool %foo() {
	%X = and bool false, undef
	ret bool %X
}
