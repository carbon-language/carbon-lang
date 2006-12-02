; RUN: llvm-upgrade < %s | llvm-as | opt -inline -disable-output

int %test() {
	unwind
}

int %caller() {
	%X = call int %test()
	ret int %X
}
