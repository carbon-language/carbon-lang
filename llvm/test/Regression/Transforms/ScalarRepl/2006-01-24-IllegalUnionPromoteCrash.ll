; RUN: llvm-as < %s | opt -scalarrepl -disable-output

target endian = big
target pointersize = 32

int %test(long %L) {
	%X = alloca int
	%Y = cast int* %X to ulong*
	store ulong 0, ulong* %Y
	%Z = load int *%X
	ret int %Z
}
