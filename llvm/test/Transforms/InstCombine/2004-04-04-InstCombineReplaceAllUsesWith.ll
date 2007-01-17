; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine -disable-output

int %test() {
	ret int 0
Loop:
	%X = add int %X, 1
	br label %Loop
}
