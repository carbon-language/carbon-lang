; RUN: llvm-as < %s | opt -inline -disable-output

int %test() {
	unwind
}

int %caller() {
	%X = call int %test()
	ret int %X
}
