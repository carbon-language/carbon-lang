; RUN: llvm-as < %s | llc

declare sbyte* %llvm.stacksave()
declare void %llvm.stackrestore(sbyte*)

int *%test(uint %N) {
	%tmp = call sbyte* %llvm.stacksave()
	%P = alloca int, uint %N
	call void %llvm.stackrestore(sbyte* %tmp)
	%Q = alloca int, uint %N
	ret int* %P
}
