; RUN: llvm-as < %s | opt -funcresolve -funcresolve | llvm-dis | not grep declare

declare void %qsortg(sbyte*, int, int)

void %test() {
	call void %qsortg(sbyte* null, int 0, int 0)
	ret void
}

int %qsortg(sbyte* %base, int %n, int %size) {
	ret int %n
}
