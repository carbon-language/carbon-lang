; Zero byte allocas should be deleted.

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep alloca

declare void %use(...)
void %test() {
	%X = alloca [0 x int]
	call void(...)* %use([0 x int] *%X)
	%Y = alloca int, uint 0
	call void(...)* %use(int* %Y)
	%Z = alloca {}
	call void(...)* %use({}* %Z)
	ret void
}
