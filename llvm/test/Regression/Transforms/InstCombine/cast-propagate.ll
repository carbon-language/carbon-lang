; RUN: llvm-as < cast-propagate.ll | opt -instcombine -mem2reg | llvm-dis | not grep load

int %test1(uint* %P) {
	%A = alloca uint 
	store uint 123, uint* %A
	%Q = cast uint* %A to int*    ; Cast the result of the load not the source
	%V = load int* %Q
	ret int %V
}
