; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | notcast

uint %testAdd(int %X, int %Y) {
	%tmp = add int %X, %Y
	%tmp.l = sext int %tmp to uint
	ret uint %tmp.l
}
