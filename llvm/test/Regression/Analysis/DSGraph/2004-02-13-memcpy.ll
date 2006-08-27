; RUN: llvm-as < %s | opt -analyze -datastructure-gc -dsgc-check-flags=Xn:SMR &&
; RUN: llvm-as < %s | opt -analyze -datastructure-gc -dsgc-check-flags=X:SMR

declare void %llvm.memcpy.i32(sbyte*, sbyte*, uint, uint)
declare void %llvm.memmove.i32(sbyte*, sbyte*, uint, uint)

void %test() {
	%X = alloca int
	%Y = alloca int
	%x = cast int* %X to sbyte*
	%y = cast int* %Y to sbyte*
	store int 4, int* %X
	call void %llvm.memcpy.i32(sbyte* %x, sbyte* %y, uint 4, uint 4)
	ret void
}

void %test2() {
	%Xn = alloca int
	%Yn = alloca int
	%xn = cast int* %Xn to sbyte*
	%yn = cast int* %Yn to sbyte*
	store int 4, int* %Xn
	call void %llvm.memmove.i32(sbyte* %xn, sbyte* %yn, uint 4, uint 4)
	ret void
}
