; RUN: analyze %s -datastructure-gc -dsgc-check-flags=X:SM

declare void %llvm.memcpy(sbyte*, sbyte*, uint, uint)

void %test() {
	%X = alloca int
	%Y = alloca int
	%x = cast int* %X to sbyte*
	%y = cast int* %Y to sbyte*
	store int 4, int* %X
	call void %llvm.memcpy(sbyte* %x, sbyte* %y, uint 4, uint 4)
	ret void
}
